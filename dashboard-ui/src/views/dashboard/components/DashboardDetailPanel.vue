<script setup lang="ts">
import { computed, h, watch, ref, toValue } from 'vue'
import type { ColumnsType } from 'ant-design-vue/es/table'
import type { TrainDetailRow } from '@/data'
import { trainMatchKey, parseTrainFlashAttnParts } from '@/features/dashboard/trainBenchmark'
import { formatCommBandwidthGb, type CommImportRow } from '@/features/dashboard/commBenchmark'
import type { BwDetailRow, BwModeKey } from '@/features/dashboard/bwBenchmark'
import { useRoute } from 'vue-router'
import DashboardVChart from '@/components/DashboardVChart.vue'
import { useEchartsGridResize } from '@/composables/useEchartsGridResize'
import {
  CI_SUMMARY,
  DIMS,
  PLATFORMS,
} from '@/data'
import { routeParamString } from '@/utils/routeParams'
import { useInfiniDashboard } from '@/composables/useInfiniDashboard'
import { useDashboardNavigation } from '@/composables/useDashboardNavigation'
import {
  computeOpRowScore,
  opRemarksContainsFailed,
  remarksExcludeIcFromScore,
} from '@/features/dashboard/operatorBenchmark'
import { BW_NVIDIA_BASELINE_GBPS, bwVsNvidiaPercent } from '@/features/dashboard/bwBenchmark'
import { scoreTierColor, SCORE_TIER_COLOR } from '@/utils/scoreColor'
import { DETAIL_DUAL_BAR_PRIMARY, DETAIL_DUAL_BAR_SECONDARY } from '@/utils/echartsInfini'

const route = useRoute()
const store = useInfiniDashboard()
const {
  currentView,
  activeDimKey,
  filterState,
  detailState,
  detailPlat,
  detailKpiCells,
  tableNotice,
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
  ciTabKey,
  setInferTab,
} = store

const { goOverview } = useDashboardNavigation()

/** 「维度名 + 详情」与 URL 维度一致；芯片名单独 tag，与 URL 平台一致 */
const detailPageTitle = computed(() => {
  if (route.name === 'detail') {
    const dk = routeParamString(route.params.dimKey)
    const dim = DIMS.find((d) => d.key === dk)
    if (dim) return `${dim.label}详情`
  }
  const dim = DIMS.find((d) => d.key === activeDimKey.value)
  return dim ? `${dim.label}详情` : '详情'
})

const detailChipNameDisplay = computed(() => {
  if (route.name === 'detail') {
    const pk = routeParamString(route.params.platKey)
    const plat = PLATFORMS.find((p) => p.key === pk)
    if (plat) return plat.name
  }
  return detailPlat.value.name
})

/** 数据明细表：除「得分」「vs NVIDIA（百分数）」列外，正文统一黑色 */
const DETAIL_TABLE_BODY_COLOR = '#000000'

/** 数据明细「对比」横条及同列数值色：与详情双柱图 DETAIL_CHART_* 一致 */
const inferMaxTps = computed(() => {
  const rows = inferDetailTabRows.value as { tps: number }[]
  const nv = inferNvidiaTabRows.value as { tps: number }[]
  const isNvidia = detailState.value.platKey === 'nvidia'
  if (isNvidia) return rows.length ? Math.max(...rows.map((r) => r.tps), 1) : 1
  return Math.max(
    rows.length ? Math.max(...rows.map((r) => r.tps)) : 0,
    nv.length ? Math.max(...nv.map((r) => r.tps || 0)) : 0,
    1,
  )
})

type InferRow = {
  configKey: string
  model: string
  batch: number
  inLen: number
  outLen?: number
  dtype?: string
  tps: number
  ttft?: number
  decodeLatencyMs?: number
  vsNvidia?: number | null
  nvidiaBaselineTps?: number | null
}

function nvInferMatch(r: InferRow) {
  return (inferNvidiaTabRows.value as InferRow[]).find((x) => x.configKey === r.configKey)
}

function inferSortTps(a: InferRow, b: InferRow) {
  return (a.tps ?? 0) - (b.tps ?? 0)
}

function inferSortModel(a: InferRow, b: InferRow) {
  return String(a.model ?? '').localeCompare(String(b.model ?? ''), 'en')
}

function inferSortBatch(a: InferRow, b: InferRow) {
  return a.batch - b.batch
}

function inferSortInLen(a: InferRow, b: InferRow) {
  return a.inLen - b.inLen
}

function inferSortOutLen(a: InferRow, b: InferRow) {
  const va = a.outLen
  const vb = b.outLen
  if (va == null && vb == null) return 0
  if (va == null) return 1
  if (vb == null) return -1
  return va - vb
}

function inferSortDtype(a: InferRow, b: InferRow) {
  return String(a.dtype ?? '').localeCompare(String(b.dtype ?? ''), 'en')
}

function inferSortTtft(a: InferRow, b: InferRow) {
  const va = a.ttft
  const vb = b.ttft
  if (va == null && vb == null) return 0
  if (va == null) return 1
  if (vb == null) return -1
  return va - vb
}

function inferSortDecodeMs(a: InferRow, b: InferRow) {
  const va = a.decodeLatencyMs
  const vb = b.decodeLatencyMs
  if (va == null && vb == null) return 0
  if (va == null) return 1
  if (vb == null) return -1
  return va - vb
}

function inferSortVsNvidia(a: InferRow, b: InferRow) {
  const va = a.vsNvidia
  const vb = b.vsNvidia
  if (va == null && vb == null) return 0
  if (va == null) return 1
  if (vb == null) return -1
  return va - vb
}

function trainSortVsA100(a: TrainDetailRow, b: TrainDetailRow) {
  return (a.vsA100 ?? 0) - (b.vsA100 ?? 0)
}

function commSortVsA100(a: CommImportRow, b: CommImportRow) {
  return (a.vsA100 ?? 0) - (b.vsA100 ?? 0)
}

type CommRowLite = { commType: string; nGpu: number; bw: number }

function nvCommMatch(r: CommRowLite) {
  return commNvidiaBaselineRows.value.find(
    (x) => x.commType === r.commType && x.nGpu === r.nGpu,
  )
}

const commBwMax = computed(() => {
  const a = commDetailRows.value.map((x) => x.bw)
  const b = commNvidiaBaselineRows.value.map((x) => x.bw)
  return Math.max(1, ...a, ...b)
})

function bwBarDenom(r: {
  add: number | null
  copy: number | null
  scale: number | null
  triad: number | null
}): number {
  const mk = bwModeKey.value
  if (mk) {
    const a = Number(r[mk] ?? 0)
    return Math.max(a, BW_NVIDIA_BASELINE_GBPS, BW_NVIDIA_BASELINE_GBPS * 0.02)
  }
  return Math.max(
    r.add ?? 0,
    r.copy ?? 0,
    r.scale ?? 0,
    r.triad ?? 0,
    BW_NVIDIA_BASELINE_GBPS,
  )
}

/** 访存表单元格：仅有 bw_GBps（avg）而四模式为空时，避免对 null 调用 toFixed */
function bwGbpsCell(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—'
  return n.toFixed(2)
}

function hasChart(opt: object) {
  return opt && Object.keys(opt).length > 0
}

const chartsGridRef = ref<HTMLElement | null>(null)
const detailLineChartRef = ref<InstanceType<typeof DashboardVChart> | null>(null)
const detailBarChartRef = ref<InstanceType<typeof DashboardVChart> | null>(null)
const detailCiChartRef = ref<InstanceType<typeof DashboardVChart> | null>(null)

const { scheduleResize: scheduleDetailTwinChartsResize } = useEchartsGridResize(
  chartsGridRef,
  () => [detailLineChartRef.value, detailBarChartRef.value],
  () => [
    activeDimKey.value,
    detailState.value.platKey,
    detailState.value.opKey,
    detailState.value.inferTab,
    toValue(lineChartOption),
    toValue(barChartOption),
  ],
)

const ciCardRef = ref<HTMLElement | null>(null)

const { scheduleResize: scheduleDetailCiChartResize } = useEchartsGridResize(
  ciCardRef,
  () => [detailCiChartRef.value],
  () => [toValue(ciChartOption), ciTabKey.value],
)

/** 从概览/对比切回详情时容器由 display:none 变为可见，需重算图表宽高 */
watch(
  currentView,
  (v) => {
    if (v !== 'detail') return
    scheduleDetailTwinChartsResize()
    scheduleDetailCiChartResize()
  },
  { flush: 'post' },
)

/** 与标题同源：详情 URL 平台优先，避免路由已切、store 尚未同步时 key/option 不一致 */
const effectiveDetailPlatKey = computed(() => {
  if (route.name === 'detail') {
    const pk = routeParamString(route.params.platKey)
    if (pk) return pk
  }
  return detailState.value.platKey
})

/** 顶栏筛选签名：切换平台后随 filterState 重建图表，避免 ECharts 合并残留 */
const detailFilterSig = computed(() => {
  const fs = filterState.value[activeDimKey.value]
  return fs ? JSON.stringify(fs) : ''
})

/** 维度/平台/筛选或视图切换时重建图表实例，避免隐藏容器内 setOption 残留上一平台图例 */
const detailTwinChartKey = computed(
  () =>
    `${currentView.value}-${activeDimKey.value}-${effectiveDetailPlatKey.value}-${detailState.value.inferTab}-${detailFilterSig.value}`,
)

/** 详情区主图标题（推理维度不用算子文案） */
const detailChartLeftTitle = computed(() => {
  switch (activeDimKey.value) {
    case 'op':
      return '延迟趋势对比'
    case 'infer':
      return 'Prefill 吞吐对比（tokens/s）'
    case 'train':
      return '训练吞吐对比（tpps）'
    case 'comm':
      return '通信带宽对比（GB/s）'
    case 'bw':
      return '访存带宽对比（GB/s）'
    default:
      return ''
  }
})

const detailChartRightTitle = computed(() => {
  switch (activeDimKey.value) {
    case 'op':
      return '各算子平均得分'
    case 'infer':
      return 'Decode 吞吐对比（tokens/s）'
    case 'train':
      return '相对 NVIDIA（%）'
    case 'comm':
      return '相对 NVIDIA（%）'
    case 'bw':
      return '相对 NVIDIA（%）'
    default:
      return ''
  }
})

function precClass(dtype: string) {
  if (dtype === 'FP16') return 'p-fp16'
  if (dtype === 'BF16') return 'p-bf16'
  return 'p-fp32'
}

/** 算子得分单元格：复用全站统一三档色阶（≥100 绿 / 60-99 橙 / <60 红） */
function scoreCellColor(score: number) {
  return scoreTierColor(score)
}

type OpDetailRow = {
  shape: string
  dtype: string
  ic: number
  pt: number
  remarks?: string
  scoreEligible?: boolean
  date?: string
}

function opRowRemarks(r: OpDetailRow) {
  return String(r.remarks ?? '')
}

function opDetailScore(r: OpDetailRow): number | null {
  return computeOpRowScore(r.ic, r.pt, opRowRemarks(r))
}

function opRowWarning(r: OpDetailRow) {
  return remarksExcludeIcFromScore(opRowRemarks(r))
}

const opLatencyMax = computed(() => {
  const rows = opDetailRows.value as OpDetailRow[]
  const m = Math.max(1e-9, ...rows.flatMap((x) => [x.ic, x.pt]))
  return m
})

/** 列宽总和与 scroll.x 一致，tableLayout=fixed 下各列更均匀 */
const opTableColumns: ColumnsType<OpDetailRow> = [
  { title: 'Shape 配置', dataIndex: 'shape', key: 'shape', width: 152, ellipsis: true },
  { title: '精度', key: 'dtype', width: 84 },
  { title: 'InfiniCore（ms）', key: 'ic', width: 128 },
  { title: 'PyTorch（ms）', key: 'pt', width: 120 },
  {
    title: '得分',
    key: 'score',
    width: 92,
    sorter: (a: OpDetailRow, b: OpDetailRow) => {
      const sa = opDetailScore(a)
      const sb = opDetailScore(b)
      if (sa == null && sb == null) return 0
      if (sa == null) return 1
      if (sb == null) return -1
      return sa - sb
    },
    sortDirections: ['descend', 'ascend'],
  },
  { title: '延迟对比', key: 'dual', width: 268, minWidth: 240 },
  { title: '备注', key: 'remarks', width: 168, ellipsis: true },
]

function opRowKey(r: OpDetailRow) {
  return `${r.shape}|${r.dtype}|${r.ic}|${r.pt}|${opRowRemarks(r)}`
}

function opCustomRow(r: OpDetailRow) {
  const warn = opRowWarning(r) && !opRemarksContainsFailed(r.remarks)
  return {
    class: warn ? 'op-detail-row op-detail-row--warn' : 'op-detail-row',
  }
}

const inferTableColumns = computed<ColumnsType<InferRow>>(() => {
  const cols: ColumnsType<InferRow> = [
    {
      title: '模型',
      dataIndex: 'model',
      key: 'model',
      minWidth: 160,
      width: 168,
      sorter: inferSortModel,
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: 'Batch',
      dataIndex: 'batch',
      key: 'batch',
      width: 84,
      sorter: inferSortBatch,
      sortDirections: ['descend', 'ascend'],
    },
    {
      title: 'In-len',
      dataIndex: 'inLen',
      key: 'inLen',
      width: 84,
      sorter: inferSortInLen,
      sortDirections: ['descend', 'ascend'],
    },
    {
      title: 'Out-len',
      dataIndex: 'outLen',
      key: 'outLen',
      width: 84,
      sorter: inferSortOutLen,
      sortDirections: ['descend', 'ascend'],
    },
    {
      title: '精度',
      dataIndex: 'dtype',
      key: 'dtype',
      width: 88,
      sorter: inferSortDtype,
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: 'TPS',
      dataIndex: 'tps',
      key: 'tps',
      width: 132,
      sorter: inferSortTps,
      sortDirections: ['descend', 'ascend'],
    },
  ]
  if (detailState.value.inferTab === 'prefill') {
    cols.push({
      title: 'TTFT',
      key: 'ttft',
      width: 100,
      sorter: inferSortTtft,
      sortDirections: ['ascend', 'descend'],
    })
  } else {
    cols.push({
      title: '单次 Decode（ms）',
      key: 'decodeLatency',
      width: 132,
      sorter: inferSortDecodeMs,
      sortDirections: ['ascend', 'descend'],
    })
  }
  if (detailState.value.platKey !== 'nvidia') {
    cols.push({
      title: 'vs NVIDIA',
      key: 'vsNvidia',
      width: 100,
      sorter: inferSortVsNvidia,
      sortDirections: ['descend', 'ascend'],
    })
  }
  cols.push({ title: '对比', key: 'compare', minWidth: 200, width: 220 })
  return cols
})

const trainTableColumns = computed<ColumnsType<TrainDetailRow>>(() => {
  const cols: ColumnsType<TrainDetailRow> = [
    { title: '框架', key: 'framework', width: 104 },
    { title: '模型', key: 'model', minWidth: 132, width: 140 },
    { title: '并行配置', key: 'parallel', width: 148 },
    { title: '参考', key: 'trainRef', width: 72 },
    { title: '精度', key: 'dtype', width: 88 },
    { title: 'Flash Attn', key: 'flashAttn', width: 116 },
    { title: '吞吐', key: 'tps', width: 132 },
    { title: '备注', key: 'note', minWidth: 120, width: 140 },
  ]
  if (detailState.value.platKey !== 'nvidia') {
    cols.push(
      { title: 'NVIDIA 基线', key: 'baseline', width: 124 },
      {
        title: 'vs NVIDIA',
        key: 'vsA100',
        width: 104,
        sorter: trainSortVsA100,
        sortDirections: ['descend', 'ascend'],
      },
    )
  }
  return cols
})

const commTableColumns = computed<ColumnsType<CommImportRow>>(() => {
  const cols: ColumnsType<CommImportRow> = [
    { title: 'Link 类型', key: 'linkType', width: 112 },
    { title: '通信类型', key: 'commType', width: 112 },
    { title: 'GPU 数', key: 'nGpu', width: 92 },
    { title: '带宽', key: 'bw', width: 104 },
    { title: '备注', key: 'note', minWidth: 120, width: 148 },
  ]
  if (detailState.value.platKey !== 'nvidia') {
    cols.push(
      { title: 'NVIDIA 基线', key: 'baseline', width: 116 },
      {
        title: 'vs NVIDIA',
        key: 'vsA100',
        width: 104,
        sorter: commSortVsA100,
        sortDirections: ['descend', 'ascend'],
      },
    )
  }
  cols.push({ title: '带宽对比', key: 'compare', minWidth: 200, width: 228 })
  return cols
})

const bwTableColumns = computed<ColumnsType<BwDetailRow>>(() => {
  const mk = bwModeKey.value
  const textCol = DETAIL_TABLE_BODY_COLOR
  const logo = detailPlat.value.logo
  const nv = bwNvidiaRefRow.value
  const emptyRowColSpan = mk ? 4 : 7
  const pendingCell = (r: BwDetailRow) => (r.avg == null ? { colSpan: 0 } : {})

  const vsPct = (record: BwDetailRow): number => {
    if (mk) {
      const v = record[mk]
      if (v != null && Number.isFinite(v)) {
        const nvV = nv?.[mk]
        if (nvV != null && Number.isFinite(nvV) && nvV > 0) {
          return Math.round((v / nvV) * 100)
        }
        return bwVsNvidiaPercent(v)
      }
    }
    return record.vsNvidia
  }

  const modeColumn = (
    mode: BwModeKey,
    opts: { firstMode: boolean },
  ): ColumnsType<BwDetailRow>[number] => ({
    title: `${mode} GB/s`,
    key: mode,
    width: 110,
    customCell: (r: BwDetailRow) =>
      r.avg == null ? (opts.firstMode ? { colSpan: emptyRowColSpan } : { colSpan: 0 }) : {},
    customRender: ({ record }) => {
      if (record.avg == null) {
        return opts.firstMode
          ? h('span', { style: { color: textCol, fontStyle: 'italic' } }, '数据待补充')
          : null
      }
      return h('span', { style: { color: DETAIL_DUAL_BAR_PRIMARY, fontWeight: 700 } }, bwGbpsCell(record[mode]))
    },
  })

  const modeCols: ColumnsType<BwDetailRow> = mk
    ? [modeColumn(mk, { firstMode: true })]
    : (['add', 'copy', 'scale', 'triad'] as const).map((mode, i) =>
        modeColumn(mode, { firstMode: i === 0 }),
      )

  const tailCols: ColumnsType<BwDetailRow> = [
    {
      title: '均值 GB/s',
      key: 'avg',
      width: 110,
      align: 'right',
      customCell: pendingCell,
      customRender: ({ record }) =>
        record.avg == null
          ? null
          : h('span', { style: { color: DETAIL_DUAL_BAR_PRIMARY, fontWeight: 700 } }, bwGbpsCell(record.avg)),
    },
    {
      title: mk ? `vs NVIDIA（${mk}）` : 'vs NVIDIA',
      key: 'vsNvidia',
      width: mk ? 128 : 110,
      customCell: pendingCell,
      sorter: (a: BwDetailRow, b: BwDetailRow) => {
        if (a.avg == null && b.avg == null) return 0
        if (a.avg == null) return -1
        if (b.avg == null) return 1
        return vsPct(a) - vsPct(b)
      },
      sortDirections: ['descend', 'ascend'],
      customRender: ({ record }) => {
        if (record.avg == null) return null
        const pct = vsPct(record)
        return h(
          'span',
          {
            style: {
              fontWeight: 700,
              color: scoreTierColor(pct),
            },
          },
          `${pct}%`,
        )
      },
    },
    {
      title: '带宽对比',
      key: 'dual',
      width: 260,
      customCell: pendingCell,
      customRender: ({ record }) => {
        if (record.avg == null) return null
        const denom = bwBarDenom(record)
        const platVal =
          mk && record[mk] != null && Number.isFinite(record[mk])
            ? Number(record[mk])
            : Number(record.avg ?? 0)
        const nvShow =
          mk != null
            ? BW_NVIDIA_BASELINE_GBPS
            : nv?.avg != null && Number.isFinite(nv.avg)
              ? Number(nv.avg)
              : BW_NVIDIA_BASELINE_GBPS
        const platW = Math.round(((platVal / denom) || 0) * 100) + '%'
        const nvW = Math.round(((nvShow / denom) || 0) * 100) + '%'
        const platMs = Number.isFinite(platVal) ? Math.round(platVal).toString() : '—'
        const nvMs = Number.isFinite(nvShow) ? nvShow.toFixed(1) : BW_NVIDIA_BASELINE_GBPS.toFixed(1)
        return h('div', { class: 'dual-bar' }, [
          h('div', { class: 'dual-row' }, [
            h('span', { class: 'dual-lbl dual-lbl--bw', style: { color: DETAIL_DUAL_BAR_PRIMARY } }, logo),
            h('div', { class: 'dual-track' }, [
              h('div', {
                class: 'dual-fill',
                style: { width: platW, background: DETAIL_DUAL_BAR_PRIMARY },
              }),
            ]),
            h('span', { class: 'dual-ms', style: { color: DETAIL_DUAL_BAR_PRIMARY } }, platMs),
          ]),
          h('div', { class: 'dual-row' }, [
            h('span', { class: 'dual-lbl dual-lbl--bw', style: { color: DETAIL_DUAL_BAR_SECONDARY } }, 'NVIDIA 基线'),
            h('div', { class: 'dual-track' }, [
              h('div', {
                class: 'dual-fill',
                style: { width: nvW, background: DETAIL_DUAL_BAR_SECONDARY },
              }),
            ]),
            h('span', { class: 'dual-ms', style: { color: DETAIL_DUAL_BAR_SECONDARY } }, nvMs),
          ]),
        ])
      },
    },
  ]

  return [
    {
      title: '型号',
      key: 'model',
      dataIndex: 'model',
      width: 120,
      customRender: ({ record }) =>
        h('span', { style: { color: textCol } }, record.model),
    },
    ...modeCols,
    ...tailCols,
  ]
})

function bwRowKey(r: BwDetailRow, i?: number) {
  const mk = bwModeKey.value ?? 'all'
  return `${mk}-${i ?? 0}-${r.model}`
}

function trainRowKey(r: TrainDetailRow, i?: number) {
  return `${trainMatchKey(r.framework, r.model, r.nGpu ?? 0, r.seqLen ?? 0, r.dtype)}-${i ?? 0}`
}

function commRowKey(r: CommImportRow) {
  return `${r.commType}-${r.nGpu}`
}

function inferRowKey(r: InferRow) {
  return r.configKey
}
</script>

<template>
  <div class="detail-panel">
    <header class="detail-panel__header">
      <div class="detail-title-row">
        <div class="detail-title-group">
          <span class="detail-title">{{ detailPageTitle }}</span>
          <span class="detail-chip-tag">{{ detailChipNameDisplay }}</span>
        </div>
        <button type="button" class="detail-back-btn" @click="goOverview">返回概览</button>
      </div>
    </header>
    <div class="detail-panel__scroll">
      <!-- 测试环境横条：随详情正文滚动（各维度 n_gpu · date · device） -->
      <div class="detail-test-env-bar">
        <span class="detail-test-env-bar__left">
          <span class="detail-test-env-bar__title">测试环境</span>
          <span class="detail-test-env-bar__sep">|</span>
          <span class="detail-test-env-bar__line">{{ detailTestEnvLine }}</span>
        </span>
        <span class="detail-test-env-bar__source">{{ detailTestEnvSourceHint }}</span>
      </div>
      <div class="kpi-grid">
        <div
          v-for="(cell, idx) in detailKpiCells"
          :key="idx"
          class="kpi-card"
        >
          <div class="kpi-val">
            {{ cell.val }}
          </div>
          <div class="kpi-lbl">{{ cell.lbl }}</div>
          <div class="kpi-sub">{{ cell.sub }}</div>
        </div>
      </div>

      <!-- 数据明细：用 flex gap 分隔，避免 margin 透出右侧栏灰底 -->
      <div class="detail-env-table-gap">
        <!-- 详情内算子筛选在 HTML 中为 display:none，此处不渲染 -->

        <div class="table-card">
      <div class="table-head-row">
        <div class="table-title">数据明细</div>
        <div v-if="activeDimKey === 'infer'" class="infer-throughput-tab-row">
          <button
            type="button"
            class="detail-back-btn infer-throughput-btn"
            :class="{ 'infer-throughput-btn--inactive': detailState.inferTab !== 'prefill' }"
            @click="setInferTab('prefill')"
          >
            Prefill 吞吐量
          </button>
          <button
            type="button"
            class="detail-back-btn infer-throughput-btn"
            :class="{ 'infer-throughput-btn--inactive': detailState.inferTab !== 'decode' }"
            @click="setInferTab('decode')"
          >
            Decode 吞吐量
          </button>
        </div>
      </div>
      <div class="detail-table-notice">
        <div class="detail-table-notice__label">得分说明：</div>
        <div class="detail-table-notice__body">{{ tableNotice }}</div>
      </div>

      <!-- 数据明细 -->
      <div class="table-wrap">
        <!-- 算子 -->
        <template v-if="activeDimKey === 'op'">
          <a-table
            v-if="opDetailRows.length"
            class="op-detail-table"
            :columns="opTableColumns"
            :data-source="(opDetailRows as OpDetailRow[])"
            :pagination="false"
            :bordered="false"
            table-layout="fixed"
            :scroll="{ x: 1062 }"
            :row-key="opRowKey"
            :custom-row="opCustomRow"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'shape'">
                <div class="op-shape-cell-wrap" :title="(record as OpDetailRow).shape">
                  <span class="op-shape-lead">
                    <span
                      v-if="opRemarksContainsFailed((record as OpDetailRow).remarks)"
                      class="op-failed-lead-icon"
                      title="remarks 含 failed"
                      aria-hidden="true"
                      >⚠</span
                    >
                    <span class="shape-cell">{{ (record as OpDetailRow).shape }}</span>
                  </span>
                </div>
              </template>
              <template v-else-if="column.key === 'dtype'">
                <span class="prec-badge" :class="precClass(record.dtype)">{{ record.dtype }}</span>
              </template>
              <template v-else-if="column.key === 'ic'">
                <span :style="{ color: DETAIL_DUAL_BAR_PRIMARY, fontWeight: 700 }">{{ record.ic.toFixed(4) }}ms</span>
              </template>
              <template v-else-if="column.key === 'pt'">
                <span :style="{ color: DETAIL_DUAL_BAR_SECONDARY, fontWeight: 700 }">{{ record.pt.toFixed(4) }}ms</span>
              </template>
              <template v-else-if="column.key === 'score'">
                <div class="score-cell">
                  <template v-if="opDetailScore(record as OpDetailRow) != null">
                    <span
                      class="score-num"
                      :style="{
                        color: scoreCellColor(Math.round(opDetailScore(record as OpDetailRow)!)),
                      }"
                    >
                      {{ Math.round(opDetailScore(record as OpDetailRow)!) }}
                    </span>
                    <span
                      :class="
                        Math.round(opDetailScore(record as OpDetailRow)!) >= 100
                          ? 'score-up'
                          : 'score-dn'
                      "
                    >
                      {{ Math.round(opDetailScore(record as OpDetailRow)!) >= 100 ? '↑' : '↓' }}
                    </span>
                  </template>
                  <template v-else>
                    <span class="score-num score-na">—</span>
                  </template>
                </div>
              </template>
              <template v-else-if="column.key === 'remarks'">
                <span class="op-remarks-cell" :title="opRowRemarks(record as OpDetailRow)">
                  {{ opRowRemarks(record as OpDetailRow) || '—' }}
                </span>
              </template>
              <template v-else-if="column.key === 'dual'">
                <div class="dual-bar">
                  <div class="dual-row">
                    <span class="dual-lbl" :style="{ color: DETAIL_DUAL_BAR_PRIMARY }">InfiniCore</span>
                    <div class="dual-track">
                      <div
                        class="dual-fill"
                        :style="{
                          width: Math.round(((record.ic as number) / opLatencyMax) * 100) + '%',
                          background: DETAIL_DUAL_BAR_PRIMARY,
                        }"
                      />
                    </div>
                    <span class="dual-ms" :style="{ color: DETAIL_DUAL_BAR_PRIMARY }">{{ record.ic.toFixed(4) }}ms</span>
                  </div>
                  <div class="dual-row">
                    <span class="dual-lbl" :style="{ color: DETAIL_DUAL_BAR_SECONDARY }">PyTorch</span>
                    <div class="dual-track">
                      <div
                        class="dual-fill"
                        :style="{
                          width: Math.round(((record.pt as number) / opLatencyMax) * 100) + '%',
                          background: DETAIL_DUAL_BAR_SECONDARY,
                        }"
                      />
                    </div>
                    <span class="dual-ms" :style="{ color: DETAIL_DUAL_BAR_SECONDARY }">{{ record.pt.toFixed(4) }}ms</span>
                  </div>
                </div>
              </template>
            </template>
          </a-table>
          <p v-else style="text-align: center; padding: 30px; color: #aaa">暂无该条件数据</p>
        </template>

        <!-- 推理 -->
        <template v-else-if="activeDimKey === 'infer'">
          <a-table
            v-if="inferDetailTabRows.length"
            :columns="inferTableColumns"
            :data-source="(inferDetailTabRows as InferRow[])"
            :pagination="false"
            :bordered="false"
            table-layout="fixed"
            :scroll="{ x: 1080 }"
            :row-key="inferRowKey"
          >
            <template #bodyCell="{ column, record, text }">
              <template v-if="column.key === 'model'">
                <span style="font-family: monospace" :style="{ color: DETAIL_TABLE_BODY_COLOR }">{{ record.model }}</span>
              </template>
              <template v-else-if="column.key === 'batch'">{{ text }}</template>
              <template v-else-if="column.key === 'inLen'">{{ text }}</template>
              <template v-else-if="column.key === 'outLen'">
                {{ record.outLen ?? '—' }}
              </template>
              <template v-else-if="column.key === 'dtype'">
                <span v-if="record.dtype" class="prec-badge" :class="precClass(record.dtype)">{{
                  record.dtype
                }}</span>
                <span v-else style="color: #000000">—</span>
              </template>
              <template v-else-if="column.key === 'tps'">
                <span :style="{ color: DETAIL_DUAL_BAR_PRIMARY, fontWeight: 700 }">{{
                  record.tps != null && Number.isFinite(record.tps)
                    ? record.tps.toLocaleString()
                    : '—'
                }}</span>
              </template>
              <template v-else-if="column.key === 'ttft'">
                <span style="color: #000000">{{ record.ttft ? record.ttft + 'ms' : '—' }}</span>
              </template>
              <template v-else-if="column.key === 'decodeLatency'">
                <span style="color: #000000">{{
                  record.decodeLatencyMs != null ? record.decodeLatencyMs + 'ms' : '—'
                }}</span>
              </template>
              <template v-else-if="column.key === 'vsNvidia'">
                <span
                  :style="{
                    color: scoreTierColor(record.vsNvidia),
                    fontWeight: 700,
                  }"
                >
                  {{ record.vsNvidia != null ? record.vsNvidia + '%' : '—' }}
                </span>
              </template>
              <template v-else-if="column.key === 'compare'">
                <div v-if="detailState.platKey === 'nvidia'" class="dual-bar">
                  <div class="dual-row">
                    <div class="dual-track" style="flex: 1">
                      <div
                        class="dual-fill"
                        :style="{
                          width: Math.round((record.tps / inferMaxTps) * 100) + '%',
                          background: DETAIL_DUAL_BAR_PRIMARY,
                        }"
                      />
                    </div>
                    <span class="dual-ms" :style="{ color: DETAIL_DUAL_BAR_PRIMARY }">{{
                      record.tps.toLocaleString()
                    }}</span>
                  </div>
                </div>
                <div v-else class="dual-bar">
                  <div class="dual-row">
                    <span class="dual-lbl" :style="{ color: DETAIL_DUAL_BAR_PRIMARY }">{{ detailPlat.logo }}</span>
                    <div class="dual-track">
                      <div
                        class="dual-fill"
                        :style="{
                          width: Math.round((record.tps / inferMaxTps) * 100) + '%',
                          background: DETAIL_DUAL_BAR_PRIMARY,
                        }"
                      />
                    </div>
                    <span class="dual-ms" :style="{ color: DETAIL_DUAL_BAR_PRIMARY }">{{
                      record.tps.toLocaleString()
                    }}</span>
                  </div>
                  <div v-if="nvInferMatch(record as InferRow)" class="dual-row">
                    <span class="dual-lbl" :style="{ color: DETAIL_DUAL_BAR_SECONDARY }">NV</span>
                    <div class="dual-track">
                      <div
                        class="dual-fill"
                        :style="{
                          width:
                            Math.round(
                              ((nvInferMatch(record as InferRow)!.tps || 0) / inferMaxTps) * 100,
                            ) + '%',
                          background: DETAIL_DUAL_BAR_SECONDARY,
                        }"
                      />
                    </div>
                    <span class="dual-ms" :style="{ color: DETAIL_DUAL_BAR_SECONDARY }">{{
                      (nvInferMatch(record as InferRow)!.tps ?? 0).toLocaleString()
                    }}</span>
                  </div>
                </div>
              </template>
              <template v-else>{{ text }}</template>
            </template>
          </a-table>
          <p v-else style="text-align: center; padding: 30px; color: #aaa">暂无该平台数据</p>
        </template>

        <!-- 训练 -->
        <template v-else-if="activeDimKey === 'train'">
          <a-table
            v-if="trainDetailRows.length"
            :columns="trainTableColumns"
            :data-source="trainDetailRows"
            :pagination="false"
            :bordered="false"
            table-layout="fixed"
            :scroll="{ x: 1200 }"
            :row-key="trainRowKey"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'framework'">
                <span>{{ record.framework }}</span>
              </template>
              <template v-else-if="column.key === 'model'">
                <span style="font-family: monospace" :style="{ color: DETAIL_TABLE_BODY_COLOR }">{{ record.model }}</span>
              </template>
              <template v-else-if="column.key === 'parallel'">
                {{ record.parallel }}
              </template>
              <template v-else-if="column.key === 'trainRef'">
                <span v-if="(record.microBatchSize ?? 0) > 0">{{ record.microBatchSize }}</span>
                <span v-else style="color: #000000">—</span>
              </template>
              <template v-else-if="column.key === 'dtype'">
                <span class="prec-badge" :class="precClass(record.dtype)">{{ record.dtype }}</span>
              </template>
              <template v-else-if="column.key === 'flashAttn'">
                <template v-for="p in [parseTrainFlashAttnParts(record.flashAttn)]" :key="record.flashAttn">
                  <span v-if="p.kind === 'on'" :style="{ color: SCORE_TIER_COLOR.high }">✓</span>
                  <template v-else>
                    <span :style="{ color: DETAIL_TABLE_BODY_COLOR }">off</span>
                    <span v-if="p.rest">{{ ' ' + p.rest }}</span>
                  </template>
                </template>
              </template>
              <template v-else-if="column.key === 'tps'">
                <span :style="{ color: DETAIL_DUAL_BAR_PRIMARY, fontWeight: 700 }">
                  {{ record.tps.toLocaleString() }} tpps
                </span>
              </template>
              <template v-else-if="column.key === 'note'">
                <span style="color: #000000">{{ record.note || '—' }}</span>
              </template>
              <template v-else-if="column.key === 'baseline'">
                <span :style="{ color: DETAIL_DUAL_BAR_SECONDARY, fontWeight: 700 }">{{ record.baseline.toLocaleString() }} tpps</span>
              </template>
              <template v-else-if="column.key === 'vsA100'">
                <span
                  :style="{
                    fontWeight: 700,
                    color: scoreTierColor(record.vsA100),
                  }"
                >
                  {{ record.vsA100 }}%
                </span>
              </template>
            </template>
          </a-table>
          <p v-else style="text-align: center; padding: 30px; color: #aaa">暂无该平台数据</p>
        </template>

        <!-- 通信 -->
        <template v-else-if="activeDimKey === 'comm'">
          <a-table
            v-if="commDetailRows.length"
            :columns="commTableColumns"
            :data-source="commDetailRows"
            :pagination="false"
            :bordered="false"
            table-layout="fixed"
            :scroll="{ x: 1040 }"
            :row-key="commRowKey"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'linkType'">
                <span :style="{ color: DETAIL_TABLE_BODY_COLOR }">{{ record.linkType }}</span>
              </template>
              <template v-else-if="column.key === 'commType'">
                <span class="prec-badge p-fp16">{{ record.commType }}</span>
              </template>
              <template v-else-if="column.key === 'nGpu'">{{ record.nGpu }} GPU</template>
              <template v-else-if="column.key === 'bw'">
                <span :style="{ color: DETAIL_DUAL_BAR_PRIMARY, fontWeight: 700 }">
                  {{ formatCommBandwidthGb(record.bw) }} GB/s
                </span>
              </template>
              <template v-else-if="column.key === 'note'">
                <span style="color: #000000">{{ record.note || '—' }}</span>
              </template>
              <template v-else-if="column.key === 'baseline'">
                <span :style="{ color: DETAIL_DUAL_BAR_SECONDARY, fontWeight: 700 }">{{ formatCommBandwidthGb(record.baseline) }} GB/s</span>
              </template>
              <template v-else-if="column.key === 'vsA100'">
                <span
                  :style="{
                    fontWeight: 700,
                    color: scoreTierColor(record.vsA100),
                  }"
                >
                  {{ record.vsA100 }}%
                </span>
              </template>
              <template v-else-if="column.key === 'compare'">
                <div v-if="detailState.platKey === 'nvidia'" class="dual-bar">
                  <div class="dual-row">
                    <div class="dual-track" style="flex: 1">
                      <div
                        class="dual-fill"
                        :style="{
                          width: Math.round((record.bw / commBwMax) * 100) + '%',
                          background: DETAIL_DUAL_BAR_PRIMARY,
                        }"
                      />
                    </div>
                    <span class="dual-ms" :style="{ color: DETAIL_DUAL_BAR_PRIMARY }">{{
                      formatCommBandwidthGb(record.bw)
                    }}</span>
                  </div>
                </div>
                <div v-else class="dual-bar">
                  <div class="dual-row">
                    <span class="dual-lbl" :style="{ color: DETAIL_DUAL_BAR_PRIMARY }">{{ detailPlat.logo }}</span>
                    <div class="dual-track">
                      <div
                        class="dual-fill"
                        :style="{
                          width: Math.round((record.bw / commBwMax) * 100) + '%',
                          background: DETAIL_DUAL_BAR_PRIMARY,
                        }"
                      />
                    </div>
                    <span class="dual-ms" :style="{ color: DETAIL_DUAL_BAR_PRIMARY }">{{
                      formatCommBandwidthGb(record.bw)
                    }}</span>
                  </div>
                  <div v-if="nvCommMatch(record as CommRowLite)" class="dual-row">
                    <span class="dual-lbl" :style="{ color: DETAIL_DUAL_BAR_SECONDARY }">NV</span>
                    <div class="dual-track">
                      <div
                        class="dual-fill"
                        :style="{
                          width:
                            Math.round(
                              ((nvCommMatch(record as CommRowLite)!.bw || 0) / commBwMax) * 100,
                            ) + '%',
                          background: DETAIL_DUAL_BAR_SECONDARY,
                        }"
                      />
                    </div>
                    <span class="dual-ms" :style="{ color: DETAIL_DUAL_BAR_SECONDARY }">{{
                      formatCommBandwidthGb(nvCommMatch(record as CommRowLite)!.bw)
                    }}</span>
                  </div>
                </div>
              </template>
            </template>
          </a-table>
          <p v-else style="text-align: center; padding: 30px; color: #aaa">暂无该平台数据</p>
        </template>

        <!-- 访存 -->
        <template v-else-if="activeDimKey === 'bw'">
          <a-table
            v-if="bwDetailRows.length"
            :key="`bw-${bwModeKey ?? 'all'}`"
            :columns="bwTableColumns"
            :data-source="bwDetailRows"
            :pagination="false"
            :bordered="false"
            :scroll="{ x: 'max-content' }"
            :row-key="bwRowKey"
          />
          <p v-else style="text-align: center; padding: 30px; color: #aaa">暂无该平台数据</p>
        </template>
      </div>
        </div>
      </div>

    <!-- 主图：仅详情视图挂载，离开概览时销毁实例，避免 display:none 下残留图例 -->
    <div v-if="currentView === 'detail'" ref="chartsGridRef" class="charts-grid">
      <div class="chart-card">
        <div class="chart-title">{{ detailChartLeftTitle }}</div>
        <DashboardVChart
          v-if="hasChart(lineChartOption)"
          :key="`line-${detailTwinChartKey}`"
          ref="detailLineChartRef"
          class="detail-chart"
          :option="lineChartOption"
          not-merge
        />
      </div>
      <div class="chart-card">
        <div class="chart-title">{{ detailChartRightTitle }}</div>
        <DashboardVChart
          v-if="hasChart(barChartOption)"
          :key="`bar-${detailTwinChartKey}`"
          ref="detailBarChartRef"
          class="detail-chart"
          :option="barChartOption"
          not-merge
        />
      </div>
    </div>

    <!-- CI -->
    <div ref="ciCardRef" class="ci-card">
      <a-tabs v-model:activeKey="ciTabKey" class="ci-detail-tabs">
        <a-tab-pane key="ci-stats" tab="CI 运行统计">
          <div class="ci-stats-grid">
            <div class="ci-stat">
              <div class="ci-stat-val">{{ CI_SUMMARY.runCount }}</div>
              <div class="ci-stat-lbl">CI 运行次数</div>
            </div>
            <div class="ci-stat">
              <div class="ci-stat-val" :style="{ color: SCORE_TIER_COLOR.high }">{{ CI_SUMMARY.avgSuccessRate }}</div>
              <div class="ci-stat-lbl">平均成功率</div>
            </div>
            <div class="ci-stat">
              <div class="ci-stat-val" :style="{ color: SCORE_TIER_COLOR.high }">{{ CI_SUMMARY.last10SuccessRate }}</div>
              <div class="ci-stat-lbl">最近10次成功率</div>
            </div>
            <div class="ci-stat">
              <div class="ci-stat-val" :style="{ color: SCORE_TIER_COLOR.low }">{{ CI_SUMMARY.failureCount }}</div>
              <div class="ci-stat-lbl">失败用例</div>
            </div>
          </div>
          <DashboardVChart
            v-if="hasChart(ciChartOption)"
            ref="detailCiChartRef"
            class="ci-chart"
            :option="ciChartOption"
            not-merge
          />
        </a-tab-pane>
        <a-tab-pane key="dispatcher" tab="Dispatcher 汇总">
          <p style="text-align: center; padding: 30px; color: #aaa; font-size: 13px">
            Dispatcher 汇总记录 — 接入真实 CI 数据后展示
          </p>
        </a-tab-pane>
        <a-tab-pane key="ci-detail" tab="CI 详细记录">
          <p style="text-align: center; padding: 30px; color: #aaa; font-size: 13px">
            CI 详细记录 — 可跳转至独立页面查看
          </p>
        </a-tab-pane>
        <a-tab-pane key="failure" tab="失败详情">
          <p style="text-align: center; padding: 30px; color: #aaa; font-size: 13px">
            失败详情 — 可跳转至独立页面查看
          </p>
        </a-tab-pane>
      </a-tabs>
    </div>
    </div>
  </div>
</template>

<style scoped>
.detail-panel {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  height: 100%;
  /* 与外层白卡片一致，避免块间距透出右侧栏灰底 */
  background-color: #fff;
}
.detail-panel__header {
  flex-shrink: 0;
}
.detail-panel__scroll {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  overflow-x: hidden;
  padding-bottom: 4px;
  background-color: #fff;
}
/* 测试环境条：与筛选项 pill 同高；置于滚动区内，随页面滚动 */
.detail-panel__scroll .detail-test-env-bar {
  box-sizing: border-box;
  margin-bottom: 13px;
  height: 28px;
  padding: 0 14px;
  background: rgb(125, 134, 173);
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 13px;
  font-size: 16px;
  font-weight: 600;
  line-height: 26px;
  color: #fff;
}
.detail-test-env-bar__left {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  flex: 1;
}
.detail-test-env-bar__title {
  flex-shrink: 0;
  font-weight: 600;
  color: #fff;
}
.detail-test-env-bar__sep {
  flex-shrink: 0;
  color: rgba(255, 255, 255, 0.55);
  font-weight: 300;
}
.detail-test-env-bar__line {
  min-width: 0;
  font-weight: 600;
  color: #fff;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.detail-test-env-bar__source {
  flex-shrink: 0;
  max-width: 46%;
  min-width: 0;
  font-size: 16px;
  font-weight: 600;
  line-height: 26px;
  color: rgba(255, 255, 255, 0.92);
  text-align: right;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.detail-panel__scroll .kpi-grid {
  margin-bottom: 28px;
}
.detail-env-table-gap {
  display: flex;
  flex-direction: column;
  gap: 13px;
  background-color: #ffffff;
}
/* 详情页内：与全局间距对齐；数据明细卡片去框线、去内边距 */
.detail-panel__scroll .table-card {
  margin-bottom: 28px;
  border: none;
  padding: 0;
}
.detail-panel__scroll .table-head-row {
  margin-bottom: 1px;
}
.detail-panel__scroll .charts-grid {
  gap: 18px;
}
.detail-panel__scroll .ci-stats-grid {
  gap: 13px;
}
/* 详情：折线柱图 / CI 去掉外框与卡片内边距，与测试环境横条同宽对齐 */
.detail-panel__scroll .chart-card,
.detail-panel__scroll .ci-card {
  border: none;
  padding: 0;
}
/* 数据明细上方：左「得分说明」、右正文，两列 */
.detail-table-notice {
  display: grid;
  grid-template-columns: max-content 1fr;
  column-gap: 0;
  align-items: start;
  margin-bottom: 8px;
  font-size: 14px;
  line-height: 1.55;
  color: #8c8c8c;
}
.detail-table-notice__label {
  white-space: nowrap;
  font-weight: 500;
  color: #737373;
  margin: 0;
  padding: 0;
}
.detail-table-notice__body {
  min-width: 0;
  margin: 0;
  padding: 0;
}
/* 推理：与「数据明细」标题同行右侧；与 .detail-back-btn 同形，未选态略淡 */
.infer-throughput-tab-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
  flex-shrink: 0;
}
.infer-throughput-btn.infer-throughput-btn--inactive {
  opacity: 0.55;
}
.infer-throughput-btn.infer-throughput-btn--inactive:hover {
  opacity: 0.72;
}
/* 详情表格：表头与表体单元格一律左对齐 */
.detail-panel__scroll :deep(.ant-table-thead > tr > th.ant-table-cell),
.detail-panel__scroll :deep(.ant-table-tbody > tr > td.ant-table-cell) {
  text-align: left !important;
}
/* 详情：数据明细标题与详情页主标题（.detail-title）同字号、字重、颜色、行高 */
.detail-panel__scroll .table-title {
  font-size: 16px;
  font-weight: 700;
  color: rgb(13, 21, 60);
  line-height: 26px;
}
/* 详情双图：折线/柱图标题同上；略收标题与卡片底留白 */
.detail-panel__scroll .charts-grid .chart-title {
  font-size: 16px;
  font-weight: 700;
  color: rgb(13, 21, 60);
  line-height: 26px;
  margin-bottom: 8px;
}
.detail-panel__scroll .charts-grid .chart-card {
  padding: 0;
}
.detail-chart {
  height: 280px;
  width: 100%;
  display: block;
  line-height: 0;
}
.ci-chart {
  height: 220px;
  width: 100%;
  display: block;
  line-height: 0;
}

/* CI 区块 Tabs：与详情主标题 .detail-title 同字号、字重、颜色（覆盖主题蓝） */
.detail-panel__scroll .ci-detail-tabs :deep(.ant-tabs-tab) {
  font-size: 16px;
  font-weight: 700;
  line-height: 26px;
  color: rgb(13, 21, 60) !important;
  padding: 10px 0;
}
.detail-panel__scroll .ci-detail-tabs :deep(.ant-tabs-tab-btn) {
  color: rgb(13, 21, 60) !important;
}
.detail-panel__scroll .ci-detail-tabs :deep(.ant-tabs-tab.ant-tabs-tab-active .ant-tabs-tab-btn) {
  color: rgb(13, 21, 60) !important;
  text-shadow: none;
}
.detail-panel__scroll .ci-detail-tabs :deep(.ant-tabs-tab:hover .ant-tabs-tab-btn) {
  color: rgb(13, 21, 60) !important;
}
.detail-panel__scroll .ci-detail-tabs :deep(.ant-tabs-ink-bar) {
  background: rgb(13, 21, 60);
}

/* 算子表首列（Shape）：固定列宽内单行省略，避免字符被硬裁切 */
.detail-panel__scroll :deep(.op-detail-table .ant-table-tbody > tr > td:first-child.ant-table-cell) {
  overflow: hidden;
  vertical-align: middle;
}

.op-shape-cell-wrap {
  min-width: 0;
  width: 100%;
  overflow: hidden;
}
.op-shape-lead {
  display: flex;
  align-items: center;
  gap: 4px;
  min-width: 0;
  width: 100%;
}
.op-shape-lead .shape-cell {
  flex: 1 1 0;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.op-failed-lead-icon {
  flex-shrink: 0;
  color: #e6a100;
  font-size: 15px;
  line-height: 1;
  cursor: help;
}

/** 算子表「备注」：最多两行，超出省略号 */
.op-remarks-cell {
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  overflow: hidden;
  text-overflow: ellipsis;
  word-break: break-word;
  line-height: 1.4;
  font-size: 11px;
  color: #000000;
  white-space: normal;
}

</style>
