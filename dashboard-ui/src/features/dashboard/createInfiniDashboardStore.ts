import { Modal } from 'ant-design-vue'
import { computed, ref } from 'vue'
import {
  BENCHMARK_DATA_META,
  BW_TABLE,
  CARD_DATA,
  COMM_TABLE,
  DIMS,
  getCiSeriesForDim,
  INFER_TABLE,
  OP_TABLE,
  PLATFORMS,
  TRAIN_TABLE,
  type BwDetailRow,
  type CommDetailRow,
  type TrainDetailRow,
} from '@/data'
import {
  applyCardFilter,
  avgOpScore,
  overlayBwOverviewCardFromFilters,
  overlayCommOverviewCardFromFilters,
  overlayInferOverviewCardFromFilters,
  overlayOpOverviewCardFromFilters,
  overlayTrainOverviewCardFromFilters,
  parseLatencyMs,
  type CardRow,
  type OpTableForOverlay,
} from '@/features/dashboard/dashboardFilterHelpers'
import {
  canComputeOpRowScore,
  computeOpRowScore,
  formatOpLatencyMs,
} from '@/features/dashboard/operatorBenchmark'
import {
  alignInferBarByBatchIn,
  filterInferRows,
  type InferTablePack,
} from '@/features/dashboard/inferBenchmark'
import {
  filterCommRows,
  formatCommBandwidthGb,
  commNvidiaBaselineBw,
  commVsPercent,
  normalizeCommType,
} from '@/features/dashboard/commBenchmark'
import { BW_NVIDIA_BASELINE_GBPS, bwVsNvidiaPercent, pickBestBwRow, pickBestBwRowByMode, type BwModeKey } from '@/features/dashboard/bwBenchmark'
import { filterTrainRows } from '@/features/dashboard/trainBenchmark'
import {
  buildBwTestEnvLine,
  buildCommTestEnvLine,
  buildInferTestEnvLine,
  buildOpTestEnvLine,
  buildTrainTestEnvLine,
  DETAIL_TEST_ENV_SOURCE_HINT,
} from '@/features/dashboard/detailTestEnvBar'
import {
  buildBwBarAvg,
  buildBwBarModes,
  buildCiLineOption,
  buildCommBarBw,
  buildCommBarVs,
  buildCompareLatencyBar,
  buildCompareScoreBar,
  buildInferDecodeBarAligned,
  buildInferPrefillBarAligned,
  buildOpBarAvgOption,
  buildOpLineOption,
  opDetailTwinGridBottom,
  buildTrainBarThroughput,
  buildTrainBarVs,
  maxCompactDetailBarGridBottom,
} from '@/utils/echartsInfini'

export type MainView = 'overview' | 'detail' | 'compare'

const OTABLE = OP_TABLE as Record<
  string,
  Record<
    string,
    {
      shape: string
      dtype: string
      ic: number
      pt: number
      remarks?: string
      scoreEligible?: boolean
      date?: string
    }[]
  >
>

function firstOpKeyForPlat(platKey: string): string {
  const keys = Object.keys(OTABLE[platKey] || {})
  return keys[0] ?? 'CausalSoftmax'
}

type InferRowCore = {
  configKey: string
  batch: number
  inLen: number
  outLen: number
  model: string
  dtype?: string
  tps: number
  ttft?: number
  decodeLatencyMs?: number
  vsNvidia?: number | null
  nvidiaBaselineTps?: number | null
  nGpu?: number
  remarks?: string
  date?: string
}

const ITABLE = INFER_TABLE as unknown as Record<
  string,
  { prefill: InferRowCore[]; decode: InferRowCore[] } | undefined
>

type TrainRow = TrainDetailRow

const TTRAIN = TRAIN_TABLE as Record<string, TrainRow[] | undefined>

const CTABLE = COMM_TABLE as Record<string, CommDetailRow[] | undefined>

const BTABLE = BW_TABLE as Record<string, BwDetailRow[] | undefined>

/** 访存详情左右柱图：与折线/柱图类目一致，用于统一 grid.bottom */
function bwTwinBarCategoryLists(
  mk: BwModeKey | null,
  rows: BwDetailRow[],
): { left: string[]; right: string[] } | null {
  if (mk) {
    const mapped = rows.filter((r) => r[mk] != null && Number.isFinite(r[mk] as number))
    if (!mapped.length) return null
    return { left: mapped.map((r) => r.model), right: [mk] }
  }
  const filtered = rows.filter((r) => r.avg != null)
  if (!filtered.length) return null
  return { left: filtered.map((r) => r.model), right: ['add', 'copy', 'scale', 'triad'] }
}

type BenchMetaFileDates = {
  trainSourceFileDateByPlatform?: Record<string, string>
  commSourceFileDateByPlatform?: Record<string, string>
  bwSourceFileDateByPlatform?: Record<string, string>
}

function benchSourceFileDate(plat: string, dim: keyof BenchMetaFileDates): string | undefined {
  const m = BENCHMARK_DATA_META as BenchMetaFileDates
  return m[dim]?.[plat]
}

/** 各维度「已选对比」默认态（行业基线）；切换维度时恢复至此。「清空」按钮仍为 [] */
const DEFAULT_COMPARE_PLAT_KEYS: readonly string[] = ['nvidia']

type DashboardDim = (typeof DIMS)[number]

export function createInfiniDashboardStore() {
  const selectedPlatKeys = ref<string[]>(PLATFORMS.map((p) => p.key))
  const comparePlatKeys = ref<string[]>([...DEFAULT_COMPARE_PLAT_KEYS])
  const currentView = ref<MainView>('overview')
  const activeDim = ref(0)
  const filterState = ref<Record<string, Record<number, number> | undefined>>({})
  const sortDesc = ref(true)
  const detailState = ref({
    platKey: 'nvidia',
    opKey: firstOpKeyForPlat('nvidia'),
    prec: '全部',
    inferTab: 'prefill' as 'prefill' | 'decode',
  })
  const bcBrand = ref('')
  const bcDim = ref('')
  const ciTabKey = ref('ci-stats')

  const activeDimKey = computed(() => DIMS[activeDim.value].key)

  /** 概览 / 对比共用：按顶栏 `filterState` 将卡片 KPI 叠成与当前筛选项一致 */
  function applyDimFilterOverlays(dim: DashboardDim, data: CardRow[]): CardRow[] {
    if (dim.key === 'op') {
      const fs = filterState.value.op || {}
      const opPill = dim.filters[0]?.pills[fs[0] ?? 0] ?? '全部'
      const dtypePill = dim.filters[1]?.pills[fs[1] ?? 0] ?? '全部'
      const hasSpecific = (fs[0] ?? 0) !== 0 || (fs[1] ?? 0) !== 0
      if (hasSpecific) {
        return data.map((c) =>
          overlayOpOverviewCardFromFilters(c, OTABLE as OpTableForOverlay, opPill, dtypePill),
        )
      }
    } else if (dim.key === 'infer') {
      const fs = filterState.value.infer || {}
      const batchPill = dim.filters[0]?.pills[fs[0] ?? 0]
      const inLenPill = dim.filters[1]?.pills[fs[1] ?? 0]
      const inferTbl = INFER_TABLE as Record<string, InferTablePack | undefined>
      return data.map((c) =>
        overlayInferOverviewCardFromFilters(c, inferTbl, batchPill, inLenPill),
      )
    } else if (dim.key === 'train') {
      const fs = filterState.value.train || {}
      const fwPill = dim.filters[0]?.pills[fs[0] ?? 0]
      const hasSpecific = (fs[0] ?? 0) !== 0
      if (hasSpecific) {
        return data.map((c) => overlayTrainOverviewCardFromFilters(c, TTRAIN, fwPill))
      }
    } else if (dim.key === 'comm') {
      const fs = filterState.value.comm || {}
      const typePill = dim.filters[0]?.pills[fs[0] ?? 0]
      return data.map((c) => overlayCommOverviewCardFromFilters(c, CTABLE, typePill))
    } else if (dim.key === 'bw') {
      const fs = filterState.value.bw || {}
      const modePill = dim.filters[0]?.pills[fs[0] ?? 0]
      const hasSpecific = (fs[0] ?? 0) !== 0
      if (hasSpecific) {
        return data.map((c) => overlayBwOverviewCardFromFilters(c, BTABLE, modePill))
      }
    }
    return data
  }

  const overviewCards = computed(() => {
    const dim = DIMS[activeDim.value]
    const raw = (CARD_DATA as Record<string, CardRow[]>)[dim.key] || []
    let data = raw.filter((c) => selectedPlatKeys.value.includes(c.key))
    data = applyDimFilterOverlays(dim, data)
    data = applyCardFilter(data, activeDim.value, filterState.value)
    const sorted = [...data].sort((a, b) => {
      const va = a.ownScore ?? 0
      const vb = b.ownScore ?? 0
      const primary = sortDesc.value ? vb - va : va - vb
      if (primary !== 0) return primary
      return String(a.key).localeCompare(String(b.key))
    })
    return sorted
  })

  const compareCards = computed(() => {
    const dim = DIMS[activeDim.value]
    const raw = (CARD_DATA as Record<string, CardRow[]>)[dim.key] || []
    let data = raw.filter((c) => comparePlatKeys.value.includes(c.key))
    data = applyDimFilterOverlays(dim, data)
    return applyCardFilter(data, activeDim.value, filterState.value)
  })

  /** 对比明细「NVIDIA基线值」列：与 compareCards 同筛选口径，不依赖是否勾选 NVIDIA 参与对比 */
  const compareNvidiaBaselineCard = computed(() => {
    if (activeDimKey.value === 'op') return undefined
    const dim = DIMS[activeDim.value]
    const raw = (CARD_DATA as Record<string, CardRow[]>)[dim.key] || []
    let data = raw.filter((c) => c.key === 'nvidia')
    if (!data.length) return undefined
    data = applyDimFilterOverlays(dim, data)
    return applyCardFilter(data, activeDim.value, filterState.value)[0]
  })

  const detailPlat = computed(() => PLATFORMS.find((p) => p.key === detailState.value.platKey)!)

  const detailCard = computed(() => {
    const dim = DIMS[activeDim.value]
    const rows = (CARD_DATA as Record<string, CardRow[]>)[dim.key] || []
    return rows.find((c) => c.key === detailState.value.platKey) || ({} as CardRow)
  })

  /** 算子详情表行 */
  const opDetailRows = computed(() => {
    const platOps = OTABLE[detailState.value.platKey] || {}
    const rows = platOps[detailState.value.opKey] || []
    if (detailState.value.prec === '全部') return rows
    return rows.filter((r) => r.dtype === detailState.value.prec)
  })

  function inferPills() {
    const dim = DIMS.find((d) => d.key === 'infer')!
    const fs = filterState.value.infer || {}
    const batchPill = dim.filters[0]?.pills[fs[0] ?? 0]
    const inPill = dim.filters[1]?.pills[fs[1] ?? 0]
    return { batchPill, inPill }
  }

  const inferPrefillFiltered = computed(() => {
    const plat = detailState.value.platKey
    const { batchPill, inPill } = inferPills()
    return filterInferRows(ITABLE[plat]?.prefill || [], batchPill, inPill)
  })

  const inferDecodeFiltered = computed(() => {
    const plat = detailState.value.platKey
    const { batchPill, inPill } = inferPills()
    return filterInferRows(ITABLE[plat]?.decode || [], batchPill, inPill)
  })

  const inferNvidiaPrefillFiltered = computed(() => {
    const { batchPill, inPill } = inferPills()
    return filterInferRows(ITABLE.nvidia?.prefill || [], batchPill, inPill)
  })

  const inferNvidiaDecodeFiltered = computed(() => {
    const { batchPill, inPill } = inferPills()
    return filterInferRows(ITABLE.nvidia?.decode || [], batchPill, inPill)
  })

  const inferDetailTabRows = computed(() =>
    detailState.value.inferTab === 'prefill'
      ? inferPrefillFiltered.value
      : inferDecodeFiltered.value,
  )

  const inferNvidiaTabRows = computed(() =>
    detailState.value.inferTab === 'prefill'
      ? inferNvidiaPrefillFiltered.value
      : inferNvidiaDecodeFiltered.value,
  )

  /** Prefill / Decode 柱图共用下边距，左右绘图区底边与 X 轴对齐 */
  const inferDetailTwinGridBottom = computed(() => {
    const pk = detailState.value.platKey
    const { batchPill } = inferPills()
    const pre = filterInferRows(ITABLE[pk]?.prefill || [], batchPill, undefined)
    const nv = filterInferRows(ITABLE.nvidia?.prefill || [], batchPill, undefined)
    const de = filterInferRows(ITABLE[pk]?.decode || [], batchPill, undefined)
    const nvDe = filterInferRows(ITABLE.nvidia?.decode || [], batchPill, undefined)
    const alPre = alignInferBarByBatchIn(pre, nv)
    const alDec = alignInferBarByBatchIn(de, nvDe)
    return maxCompactDetailBarGridBottom(alPre.categories, alDec.categories)
  })

  function trainFrameworkPill() {
    const dim = DIMS.find((d) => d.key === 'train')!
    const fs = filterState.value.train || {}
    return dim.filters[0]?.pills[fs[0] ?? 0]
  }

  const trainDetailRows = computed(() => {
    const plat = detailState.value.platKey
    const pill = trainFrameworkPill()
    return filterTrainRows(TTRAIN[plat], pill)
  })

  const commNvidiaBaselineRows = computed(() => CTABLE.nvidia || [])

  const commDetailRows = computed(() => {
    const plat = detailState.value.platKey
    const dim = DIMS.find((d) => d.key === 'comm')!
    const fs = filterState.value.comm || {}
    const pill = dim.filters[0]?.pills[fs[0] ?? 0]
    return filterCommRows(CTABLE[plat], pill)
  })

  /** 访存详情表：顶栏「模式」非「全部」时仅保留该列有数值的型号行 */
  const bwModeKey = computed((): BwModeKey | null => {
    const dim = DIMS.find((d) => d.key === 'bw')!
    const fs = filterState.value.bw || {}
    const p = dim.filters[0]?.pills[fs[0] ?? 0]
    if (!p || p === '全部') return null
    const m = String(p).toLowerCase()
    if (m === 'add' || m === 'copy' || m === 'scale' || m === 'triad') return m
    return null
  })

  const bwDetailRows = computed(() => {
    const plat = detailState.value.platKey
    const rows = BTABLE[plat] || []
    const mk = bwModeKey.value
    if (!mk) return rows
    return rows.filter((r) => {
      const v = r[mk]
      return v != null && Number.isFinite(v)
    })
  })

  const bwNvidiaRefRow = computed(() => {
    const rows = BTABLE.nvidia || []
    return pickBestBwRow(rows) ?? rows[0] ?? null
  })

  const detailTestEnvLine = computed(() => {
    const dk = activeDimKey.value
    const plat = detailState.value.platKey
    if (dk === 'infer') {
      const pack = ITABLE[plat]
      const inferWide = [...(pack?.prefill ?? []), ...(pack?.decode ?? [])]
      return buildInferTestEnvLine(
        inferDetailTabRows.value as InferRowCore[],
        inferWide,
      )
    }
    if (dk === 'op') {
      const platOps = OTABLE[plat] || {}
      const platformWide = Object.values(platOps).flat() as { date?: string }[]
      return buildOpTestEnvLine(plat, opDetailRows.value, platformWide)
    }
    if (dk === 'train') {
      return buildTrainTestEnvLine(
        plat,
        trainDetailRows.value,
        TTRAIN[plat] ?? [],
        benchSourceFileDate(plat, 'trainSourceFileDateByPlatform'),
      )
    }
    if (dk === 'comm') {
      return buildCommTestEnvLine(
        plat,
        commDetailRows.value,
        CTABLE[plat] ?? [],
        benchSourceFileDate(plat, 'commSourceFileDateByPlatform'),
      )
    }
    if (dk === 'bw') {
      return buildBwTestEnvLine(
        plat,
        BTABLE[plat] || [],
        benchSourceFileDate(plat, 'bwSourceFileDateByPlatform'),
      )
    }
    return '—'
  })

  const detailTestEnvSourceHint = computed(() => DETAIL_TEST_ENV_SOURCE_HINT)

  const lineChartOption = computed(() => {
    /** 非详情视图不产出 option，配合详情图 v-if 卸载实例，避免 display:none 内 setOption 残留图例 */
    if (currentView.value !== 'detail') return {}
    const plat = detailPlat.value
    const dk = activeDimKey.value
    if (dk === 'op') {
      const rows = opDetailRows.value
      if (!rows.length) return {}
      const platOps = OTABLE[detailState.value.platKey] || {}
      const opKeys = Object.keys(platOps)
      const shapes = rows.map((r) => r.shape)
      const gridBottom = opDetailTwinGridBottom(shapes, opKeys)
      return buildOpLineOption(rows, { gridBottom })
    }
    if (dk === 'infer') {
      const pk = detailState.value.platKey
      const { batchPill } = inferPills()
      /** 柱图：仅随 Batch pill；列出该 batch 下全部 in_len（不受 In-len pill 收窄） */
      const pre = filterInferRows(ITABLE[pk]?.prefill || [], batchPill, undefined)
      const nv = filterInferRows(ITABLE.nvidia?.prefill || [], batchPill, undefined)
      if (!pre.length && !nv.length) return {}
      const al = alignInferBarByBatchIn(pre, nv)
      return buildInferPrefillBarAligned(
        al.categories,
        al.platVals,
        al.nvVals,
        plat.name,
        'NVIDIA',
        {
          gridBottom: inferDetailTwinGridBottom.value,
          omitTwinBaselineSeries: pk === 'nvidia',
        },
      )
    }
    if (dk === 'train') {
      const rows = trainDetailRows.value
      if (!rows?.length) return {}
      const categories = rows.map((r) => `${r.framework}·${r.model}`)
      const gridBottom = maxCompactDetailBarGridBottom(categories)
      return buildTrainBarThroughput(rows, {
        gridBottom,
        omitTwinBaselineSeries: detailState.value.platKey === 'nvidia',
      })
    }
    if (dk === 'comm') {
      const rows = commDetailRows.value
      if (!rows?.length) return {}
      const catsBw = rows.map((r) => r.commType + ' ' + r.nGpu + 'GPU')
      const catsVs = rows.map((r) => r.commType)
      const gridBottom = maxCompactDetailBarGridBottom(catsBw, catsVs)
      return buildCommBarBw(rows, {
        gridBottom,
        omitTwinBaselineSeries: detailState.value.platKey === 'nvidia',
      })
    }
    if (dk === 'bw') {
      const mk = bwModeKey.value
      const rows = bwDetailRows.value
      const pair = bwTwinBarCategoryLists(mk, rows)
      if (!pair) return {}
      const gridBottom = maxCompactDetailBarGridBottom(pair.left, pair.right)
      if (mk) {
        const mapped = rows
          .filter((r) => r[mk] != null && Number.isFinite(r[mk]))
          .map((r) => ({ ...r, avg: r[mk] as number }))
        if (!mapped.length) return {}
        const nv = bwNvidiaRefRow.value?.[mk]
        const baseline =
          nv != null && Number.isFinite(nv) && nv > 0 ? nv : BW_NVIDIA_BASELINE_GBPS
        return buildBwBarAvg(mapped, baseline, `${mk} GB/s`, {
          gridBottom,
          omitTwinBaselineSeries: detailState.value.platKey === 'nvidia',
        })
      }
      const filtered = rows.filter((r) => r.avg != null)
      if (!filtered.length) return {}
      return buildBwBarAvg(filtered, BW_NVIDIA_BASELINE_GBPS, '均值 GB/s', {
        gridBottom,
        omitTwinBaselineSeries: detailState.value.platKey === 'nvidia',
      })
    }
    return {}
  })

  const barChartOption = computed(() => {
    if (currentView.value !== 'detail') return {}
    const plat = detailPlat.value
    const dk = activeDimKey.value
    if (dk === 'op') {
      const platOps = OTABLE[detailState.value.platKey] || {}
      const opKeys = Object.keys(platOps)
      const scores = opKeys.map((op) => {
        const rs = platOps[op] || []
        const valid = rs.filter((r) =>
          canComputeOpRowScore(r.ic, r.pt, r.remarks ?? ''),
        )
        return valid.length
          ? Math.round(
              valid.reduce(
                (a, r) => a + (computeOpRowScore(r.ic, r.pt, r.remarks ?? '') ?? 0),
                0,
              ) / valid.length,
            )
          : 0
      })
      const rows = opDetailRows.value
      const shapes = rows.map((r) => r.shape)
      const gridBottom = opDetailTwinGridBottom(shapes, opKeys)
      return buildOpBarAvgOption(opKeys, scores, { gridBottom })
    }
    if (dk === 'infer') {
      const pk = detailState.value.platKey
      const { batchPill } = inferPills()
      const de = filterInferRows(ITABLE[pk]?.decode || [], batchPill, undefined)
      const nvDe = filterInferRows(ITABLE.nvidia?.decode || [], batchPill, undefined)
      if (!de.length && !nvDe.length) return {}
      const al = alignInferBarByBatchIn(de, nvDe)
      return buildInferDecodeBarAligned(
        al.categories,
        al.platVals,
        al.nvVals,
        plat.name,
        'NVIDIA',
        {
          gridBottom: inferDetailTwinGridBottom.value,
          omitTwinBaselineSeries: pk === 'nvidia',
        },
      )
    }
    if (dk === 'train') {
      const rows = trainDetailRows.value
      if (!rows?.length) return {}
      const categories = rows.map((r) => `${r.framework}·${r.model}`)
      const gridBottom = maxCompactDetailBarGridBottom(categories)
      return buildTrainBarVs(rows, {
        gridBottom,
        trainPlatBarName: plat.name,
        omitTwinBaselineSeries: detailState.value.platKey === 'nvidia',
      })
    }
    if (dk === 'comm') {
      const rows = commDetailRows.value
      if (!rows?.length) return {}
      const catsBw = rows.map((r) => r.commType + ' ' + r.nGpu + 'GPU')
      const catsVs = rows.map((r) => r.commType)
      const gridBottom = maxCompactDetailBarGridBottom(catsBw, catsVs)
      return buildCommBarVs(rows, {
        gridBottom,
        commPlatBarName: plat.name,
        omitTwinBaselineSeries: detailState.value.platKey === 'nvidia',
      })
    }
    if (dk === 'bw') {
      const platKey = detailState.value.platKey
      const rowsFull = BTABLE[platKey] || []
      const rowsDetail = bwDetailRows.value
      const mk = bwModeKey.value
      const pair = bwTwinBarCategoryLists(mk, rowsDetail)
      if (!pair) return {}
      const gridBottom = maxCompactDetailBarGridBottom(pair.left, pair.right)
      const nvidiaRow = bwNvidiaRefRow.value
      if (!nvidiaRow) return {}
      if (mk) {
        const rowsWithMode = rowsDetail.filter((r) => r[mk] != null && Number.isFinite(r[mk] as number))
        if (!rowsWithMode.length) return {}
        return buildBwBarModes(rowsWithMode, nvidiaRow, mk, {
          gridBottom,
          bwPlatBarName: plat.name,
          omitTwinBaselineSeries: platKey === 'nvidia',
        })
      }
      const best = pickBestBwRowByMode(rowsFull, null)
      if (!best) return {}
      return buildBwBarModes(best, nvidiaRow, null, {
        gridBottom,
        omitTwinBaselineSeries: platKey === 'nvidia',
      })
    }
    return {}
  })

  const ciChartOption = computed(() => {
    if (currentView.value !== 'detail') return {}
    return buildCiLineOption(getCiSeriesForDim(activeDimKey.value))
  })

  const compareScoreOption = computed(() => {
    const cards = compareCards.value
    const plats = cards.map((c) => PLATFORMS.find((p) => p.key === c.key)!).filter(Boolean)
    if (!cards.length || !plats.length) return {}
    const mode = activeDimKey.value === 'op' ? 'operator' : 'platformScore'
    return buildCompareScoreBar(cards as { key: string; ownScore: number | null }[], plats, mode)
  })

  const compareLatencyOption = computed(() => {
    const cards = compareCards.value.filter((c) => parseLatencyMs(c.ownVal as string) != null)
    const names = cards.map((c) => PLATFORMS.find((p) => p.key === c.key)!.name)
    const latencies = cards.map((c) => parseLatencyMs(c.ownVal as string) as number)
    if (!latencies.length) return {}
    return buildCompareLatencyBar(names, latencies)
  })

  const detailTitle = computed(() => {
    const dim = DIMS[activeDim.value]
    return `${dim.label}详情`
  })

  const tableNotice = computed(() => {
    const k = activeDimKey.value
    /**
     * 算子维度（产品亦称「测试维度」）数据明细「得分说明」——文案已定稿。
     * 禁止在未获产品书面确认的情况下修改下列 return 字符串或删减本注释块。
     */
    if (k === 'op')
      return '每行得分 = PyTorch延迟 ÷ InfiniCore延迟 × 100（同 shape + dtype）· >100 自研更快 · 平均得分 = 所有行均值 · ✦ InfiniCore 为自研框架'
    if (k === 'infer')
      return '吞吐量（tokens/s）· 数据越高越好 · vs NVIDIA InfiniLM 同配置'
    if (k === 'train')
      return '训练吞吐 tpps = tokens per process per second · vs NVIDIA Megatron 基线'
    if (k === 'comm')
      return '带宽 vs NVIDIA NVLink 基线 · 单向带宽'
    if (k === 'bw')
      return '访存带宽（GB/s）vs NVIDIA 访存带宽 · add / copy / scale / triad 四模式均值'
    return ''
  })

  const comparePageTitle = computed(() => `${DIMS[activeDim.value].label} · 平台横向对比`)

  const compareKpiBlocks = computed(() => {
    const cards = compareCards.value
    const bestScore = Math.max(...cards.map((c) => (c.ownScore ?? 0) || 0))
    const sorted = [...cards].sort((a, b) => (b.ownScore ?? 0) - (a.ownScore ?? 0))
    return cards.map((c) => {
      const p = PLATFORMS.find((x) => x.key === c.key)!
      const rank = sorted.findIndex((x) => x.key === c.key) + 1
      const isBest = c.ownScore === bestScore
      return { card: c, plat: p, rank, isBest }
    })
  })

  const compareTableRows = computed(() => {
    const cards = compareCards.value
    const base = cards.find((x) => x.key === 'nvidia')
    return cards.map((c) => {
      const p = PLATFORMS.find((x) => x.key === c.key)!
      const delta =
        base?.ownScore != null && base.ownScore !== 0
          ? Math.round(((c.ownScore ?? 0) - base.ownScore) / base.ownScore * 100)
          : null
      return { plat: p, card: c, delta }
    })
  })

  /** 详情顶栏 4 格 KPI（与 HTML renderKPI 一致） */
  const detailKpiCells = computed(() => {
    const platKey = detailState.value.platKey
    const dimKey = activeDimKey.value
    if (dimKey === 'op') {
      const opRows = opDetailRows.value
      let bestScore = -1
      let bestRow: (typeof opRows)[0] | null = null
      for (const r of opRows) {
        const s = computeOpRowScore(r.ic, r.pt, r.remarks ?? '')
        if (s != null && s > bestScore) {
          bestScore = s
          bestRow = r
        }
      }
      const cardRow = detailCard.value as CardRow & { extra?: string; opRecordSub?: string }
      return [
        {
          val: bestScore >= 0 ? Math.round(bestScore) : '—',
          lbl: '最优得分',
          sub: 'InfiniCore vs PyTorch（全量测试中最高分所在行）',
        },
        {
          val: opRows.length || '—',
          lbl: '测试记录',
          sub: cardRow.opRecordSub || cardRow.extra || '',
        },
        {
          val: bestRow ? formatOpLatencyMs(bestRow.ic) : '—',
          lbl: 'InfiniCore 代表延迟',
          sub: '自研框架',
          valSm: true,
        },
        {
          val: bestRow ? formatOpLatencyMs(bestRow.pt) : '—',
          lbl: 'PyTorch 代表延迟',
          sub: '开源基准',
          valSm: true,
        },
      ]
    }
    const c = detailCard.value as CardRow & {
      ownVal?: string
      n?: number
      extra?: string
      ownScore?: number | null
    }
    if (dimKey === 'infer') {
      /** 与表格 / 柱状图一致：顶栏 Batch、In-len 筛选后的子集 */
      const prefillAll = inferPrefillFiltered.value
      const decodeAll = inferDecodeFiltered.value
      const preTps = prefillAll.map((r) => r.tps).filter((t) => Number.isFinite(t))
      const decTps = decodeAll.map((r) => r.tps).filter((t) => Number.isFinite(t))
      const peakPrefill = preTps.length ? Math.max(...preTps) : null
      const peakDecode = decTps.length ? Math.max(...decTps) : null
      const ttftVals = prefillAll
        .map((r) => r.ttft)
        .filter((x): x is number => x != null && Number.isFinite(x) && x > 0)
      const minTTFT = ttftVals.length ? Math.min(...ttftVals) : null
      const argPrefAll = prefillAll.filter((r) => Number.isFinite(r.tps)).length
        ? prefillAll.filter((r) => Number.isFinite(r.tps)).reduce((a, b) => (a.tps >= b.tps ? a : b))
        : null
      const model = argPrefAll?.model || prefillAll[0]?.model || c.extra || ''
      const nConfigs = new Set([
        ...prefillAll.map((r) => r.configKey),
        ...decodeAll.map((r) => r.configKey),
      ])
      const fsInf = filterState.value.infer || {}
      const inferFiltered =
        (fsInf[0] ?? 0) !== 0 || (fsInf[1] ?? 0) !== 0
      const scopeLbl = inferFiltered ? '当前筛选 · ' : '全表 · '
      return [
        {
          val: peakPrefill != null ? peakPrefill.toLocaleString() : c.ownVal || '—',
          lbl: 'Prefill 峰值 TPS',
          sub: `InfiniLM · ${model}`,
          valSm: true,
        },
        {
          val: peakDecode != null ? peakDecode.toLocaleString() : '—',
          lbl: 'Decode 峰值 TPS',
          sub: `InfiniLM · ${model}`,
          valSm: true,
        },
        {
          val: minTTFT != null ? minTTFT + 'ms' : '—',
          lbl: '最低 TTFT',
          sub: `${scopeLbl}il_ttft_ms（有效值取最小）`,
          valSm: true,
        },
        {
          val: (nConfigs.size || c.n || '—') as string | number,
          lbl: '测试配置数',
          sub: `${scopeLbl}Prefill ∪ Decode 配置键`,
        },
      ]
    }
    if (dimKey === 'train') {
      const rows = trainDetailRows.value
      const finite = (rows || []).filter((r) => Number.isFinite(r.tps))
      const best = finite.length ? finite.reduce((a, b) => (a.tps >= b.tps ? a : b)) : undefined
      return [
        {
          val: best?.tps != null ? best.tps.toLocaleString() + ' tpps' : c.ownVal || '—',
          lbl: '最高训练吞吐',
          sub: `${best?.framework || ''} · ${best?.model || ''}`,
          valSm: true,
        },
        {
          val: best?.vsA100 != null ? best.vsA100 + '%' : '—',
          lbl: 'vs NVIDIA 基线',
          sub: '同配置吞吐比',
          valSm: true,
        },
        { val: finite.length || '—', lbl: '测试条数', sub: c.extra || '' },
        {
          val: best?.parallel || '—',
          lbl: '代表配置',
          sub: best?.dtype || '',
          valSm: true,
        },
      ]
    }
    if (dimKey === 'comm') {
      // 四格 KPI：全表按类型取最高带宽代表行，不受顶栏「通信类型」子集影响（表格/图仍用 commDetailRows）
      const allRows = CTABLE[platKey] || []
      const nvRows = commNvidiaBaselineRows.value
      const pickMaxBwByType = (want: 'p2p' | 'allreduce') => {
        const list = allRows.filter((r) => normalizeCommType(r.commType) === want)
        if (!list.length) return undefined
        return list.reduce((a, b) => (a.bw >= b.bw ? a : b))
      }
      const p2p = pickMaxBwByType('p2p')
      const ar = pickMaxBwByType('allreduce')
      const p2pNvBw = p2p ? commNvidiaBaselineBw(nvRows, 'p2p', p2p.nGpu) : undefined
      const arNvBw = ar ? commNvidiaBaselineBw(nvRows, 'allreduce', ar.nGpu) : undefined
      const p2pVsPct =
        p2p && p2pNvBw != null && p2pNvBw > 0 ? commVsPercent(p2p.bw, p2pNvBw) : p2p != null ? p2p.vsA100 : null
      const arVsPct =
        ar && arNvBw != null && arNvBw > 0 ? commVsPercent(ar.bw, arNvBw) : ar != null ? ar.vsA100 : null
      return [
        {
          val: p2p ? formatCommBandwidthGb(p2p.bw) + ' GB/s' : '—',
          lbl: 'P2P 带宽',
          sub: p2p ? p2p.linkType + ' · ' + p2p.nGpu + ' GPU' : '',
          valSm: true,
        },
        {
          val: ar ? formatCommBandwidthGb(ar.bw) + ' GB/s' : '—',
          lbl: 'AllReduce 带宽',
          sub: ar ? ar.linkType + ' · ' + ar.nGpu + ' GPU' : '',
          valSm: true,
        },
        {
          val: p2pVsPct != null ? p2pVsPct + '%' : '—',
          lbl: 'P2P vs NVIDIA',
          sub: '同 comm_type + n_gpu 基线',
          valSm: true,
        },
        {
          val: arVsPct != null ? arVsPct + '%' : '—',
          lbl: 'AllReduce vs NVIDIA',
          sub: '同 comm_type + n_gpu 基线',
          valSm: true,
        },
      ]
    }
    if (dimKey === 'bw') {
      const allRows = BTABLE[platKey] || []
      const mk = bwModeKey.value
      const nvRef = bwNvidiaRefRow.value
      if (!mk) {
        const best = pickBestBwRow(allRows)
        return [
          {
            val: best?.avg != null ? best.avg.toFixed(1) + ' GB/s' : '—',
            lbl: 'HBM 均值带宽',
            sub: `${best?.model || ''} · 4模式均值`,
            valSm: true,
          },
          {
            val: best?.avg != null ? bwVsNvidiaPercent(best.avg) + '%' : '—',
            lbl: 'vs NVIDIA',
            sub: `NVIDIA 基线 ${BW_NVIDIA_BASELINE_GBPS} GB/s`,
            valSm: true,
          },
          {
            val: best?.add != null ? best.add.toFixed(1) : '—',
            lbl: 'add GB/s',
            sub: '读写混合模式',
            valSm: true,
          },
          {
            val: best?.triad != null ? best.triad.toFixed(1) : '—',
            lbl: 'triad GB/s',
            sub: '综合压力模式',
            valSm: true,
          },
        ]
      }
      const bestM = pickBestBwRowByMode(allRows, mk)
      const v = bestM?.[mk]
      const nvV = nvRef?.[mk]
      const vsPct =
        v != null && nvV != null && nvV > 0 ? Math.round((v / nvV) * 100) : null
      const nValid = bwDetailRows.value.length
      return [
        {
          val: v != null ? v.toFixed(1) + ' GB/s' : '—',
          lbl: `${mk} 带宽`,
          sub: `${bestM?.model ?? ''} · 当前模式最优`,
          valSm: true,
        },
        {
          val: vsPct != null ? vsPct + '%' : '—',
          lbl: 'vs NVIDIA',
          sub: '参考行同模式列对比',
          valSm: true,
        },
        {
          val: bestM?.avg != null ? bestM.avg.toFixed(1) + ' GB/s' : '—',
          lbl: '四模式均值',
          sub: '同行 HBM 均值',
          valSm: true,
        },
        {
          val: nValid || '—',
          lbl: '有效型号数',
          sub: `含「${mk}」实测数据`,
        },
      ]
    }
    return []
  })

  function togglePlat(key: string) {
    const i = selectedPlatKeys.value.indexOf(key)
    if (i >= 0) selectedPlatKeys.value = selectedPlatKeys.value.filter((k) => k !== key)
    else selectedPlatKeys.value = [...selectedPlatKeys.value, key]
  }

  function selectAll() {
    const allSel = PLATFORMS.every((p) => selectedPlatKeys.value.includes(p.key))
    selectedPlatKeys.value = allSel ? [] : PLATFORMS.map((p) => p.key)
  }

  function selectDomestic() {
    const domesticKeys = PLATFORMS.filter((p) => p.domestic).map((p) => p.key)
    const domesticSet = new Set(domesticKeys)
    const sel = selectedPlatKeys.value
    const isDomesticOnly =
      sel.length === domesticKeys.length && sel.every((k) => domesticSet.has(k))
    selectedPlatKeys.value = isDomesticOnly ? [] : domesticKeys
  }

  function setDim(i: number) {
    if (i !== activeDim.value) {
      resetCompareToDefault()
    }
    activeDim.value = i
    filterState.value = {}
    switchMainView('overview')
  }

  function setFilter(fi: number, pi: number) {
    const k = DIMS[activeDim.value].key
    const dim = DIMS[activeDim.value]
    const pill = dim.filters[fi]?.pills[pi]
    const prevFs = filterState.value[k] || {}
    /** 整体替换，避免嵌套原地改导致部分订阅（如表格）不刷新 */
    filterState.value = {
      ...filterState.value,
      [k]: { ...prevFs, [fi]: pi },
    }
    if (currentView.value === 'detail' && k === 'op') {
      if (fi === 0 && pill && pill !== '全部') detailState.value.opKey = pill
      if (fi === 1 && pill && pill !== '全部') detailState.value.prec = pill
    }
  }

  function toggleSort() {
    sortDesc.value = !sortDesc.value
  }

  function toggleCompare(key: string) {
    const idx = comparePlatKeys.value.indexOf(key)
    if (idx >= 0) comparePlatKeys.value = comparePlatKeys.value.filter((k) => k !== key)
    else {
      if (comparePlatKeys.value.length >= 4) {
        Modal.warning({ title: '提示', content: '最多选择 4 个平台进行对比' })
        return
      }
      comparePlatKeys.value = [...comparePlatKeys.value, key]
    }
  }

  function clearCompare() {
    comparePlatKeys.value = []
  }

  function resetCompareToDefault() {
    comparePlatKeys.value = [...DEFAULT_COMPARE_PLAT_KEYS]
  }

  function switchMainView(v: MainView) {
    currentView.value = v
  }

  function opTagKeys(): string[] {
    const platOps = OTABLE[detailState.value.platKey] || {}
    const ops = Object.keys(platOps).length ? Object.keys(platOps) : ['全部数据']
    if (!platOps[detailState.value.opKey]) detailState.value.opKey = ops[0]
    return ops
  }

  function setOp(op: string) {
    detailState.value.opKey = op
  }

  function setPrec(p: string) {
    detailState.value.prec = p
  }

  function setInferTab(t: 'prefill' | 'decode') {
    detailState.value.inferTab = t
  }

  return {
    PLATFORMS,
    DIMS,
    CARD_DATA,
    OP_TABLE: OTABLE,
    INFER_TABLE,
    TRAIN_TABLE,
    COMM_TABLE,
    BW_TABLE,
    selectedPlatKeys,
    comparePlatKeys,
    currentView,
    activeDim,
    filterState,
    sortDesc,
    detailState,
    bcBrand,
    bcDim,
    ciTabKey,
    activeDimKey,
    overviewCards,
    compareCards,
    compareNvidiaBaselineCard,
    detailPlat,
    detailCard,
    opDetailRows,
    inferDetailTabRows,
    inferNvidiaTabRows,
    detailTestEnvLine,
    detailTestEnvSourceHint,
    trainDetailRows,
    commDetailRows,
    commNvidiaBaselineRows,
    bwDetailRows,
    bwModeKey,
    bwNvidiaRefRow,
    lineChartOption,
    barChartOption,
    ciChartOption,
    compareScoreOption,
    compareLatencyOption,
    detailTitle,
    tableNotice,
    comparePageTitle,
    compareKpiBlocks,
    compareTableRows,
    detailKpiCells,
    avgOpScore,
    togglePlat,
    selectAll,
    selectDomestic,
    setDim,
    setFilter,
    toggleSort,
    toggleCompare,
    clearCompare,
    resetCompareToDefault,
    switchMainView,
    opTagKeys,
    setOp,
    setPrec,
    setInferTab,
    parseLatencyMs,
  }
}

export type InfiniDashboardStore = ReturnType<typeof createInfiniDashboardStore>
