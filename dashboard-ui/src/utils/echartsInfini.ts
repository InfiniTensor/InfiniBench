/** ECharts option 构建 — 数值来自 @/data，与 dashboard_preview.html Chart.js 一致 */

import { CI_CHART_LABELS } from '@/data'
import { BW_NVIDIA_BASELINE_GBPS } from '@/features/dashboard/bwBenchmark'

/**
 * 详情页折线/柱图：双系列主色与图例一致（本机/实测蓝 + 基线或 PyTorch 等辅色）；
 * 与数据明细「对比」横条常量 DETAIL_DUAL_BAR_* 同源
 */
export const DETAIL_CHART_PRIMARY = '#5B9BD5'
/** 辅色（PyTorch / NVIDIA 基线等）：白底可读，略浅于纯深绿 */
export const DETAIL_CHART_SECONDARY = '#83BB46'

/** 详情数据明细「对比」双横条及同列标签/数值色（与上方双柱图系列色一致） */
export const DETAIL_DUAL_BAR_PRIMARY = '#5B9BD5'
export const DETAIL_DUAL_BAR_SECONDARY = '#83BB46'

/** 详情页图表：坐标轴刻度、轴名、图例、悬浮提示等文字统一为 14px */
export const DETAIL_CHART_AXIS_FONT_SIZE = 14

/** 详情页坐标轴刻度、轴名等纯黑（ECharts 默认偏灰） */
export const DETAIL_AXIS_TEXT_COLOR = '#000000'

const DETAIL_CHART_PRIMARY_SOFT = `${DETAIL_CHART_PRIMARY}cc`
const DETAIL_CHART_SECONDARY_SOFT = `${DETAIL_CHART_SECONDARY}cc`

/** 详情页柱状图单柱最大宽度（px） */
const DETAIL_BAR_MAX_WIDTH = 20

/** 详情各维度柱图：含 y 轴刻度与名称，左右留白避免贴轴/右侧溢出 */
const DETAIL_BAR_GRID = {
  left: 10,
  right: 22,
  containLabel: true,
} as const

const DETAIL_AXIS_TOOLTIP = {
  trigger: 'axis' as const,
  textStyle: { fontSize: DETAIL_CHART_AXIS_FONT_SIZE, color: DETAIL_AXIS_TEXT_COLOR },
}

const DETAIL_CHART_LEGEND = {
  top: 2,
  right: 8,
  textStyle: { fontSize: DETAIL_CHART_AXIS_FONT_SIZE, color: DETAIL_AXIS_TEXT_COLOR },
}

type NamedChartSeries = { name?: string }

/** 图例与 series 同名同步，避免 setOption 合并或隐藏容器更新后残留旧系列名（如 PyTorch） */
function detailChartLegend(series: NamedChartSeries[]) {
  return {
    ...DETAIL_CHART_LEGEND,
    data: series.map((s) => s.name).filter((n): n is string => Boolean(n)),
  }
}

/** 对比页柱图：略大于默认 grid.left，竖排 Y 轴名与刻度不裁切；勿过大以免左侧留白 */
const COMPARE_BAR_GRID_LEFT = 30

/** 对比页 Y 轴竖排名与刻度间距（nameGap 为轴名到轴线法向距离，过小会贴数字） */
const COMPARE_Y_NAME_GAP_SCORE = 48
const COMPARE_Y_NAME_GAP_LATENCY = 56

/** 算子详情页「延迟趋势」折线 +「各算子平均得分」柱图：统一 grid，使绘图区垂直范围一致 */
const OP_DETAIL_TWIN_GRID_TOP = 54

/**
 * 详情页类目轴（≥3 列）：横排 + 过长省略（`overflow: 'truncate'` + `width`）；≤2 列不截断，由 detailCategoryAxisLabel 切换
 */
export const DETAIL_DIM_BAR_GRID_BOTTOM = 14
export const DETAIL_DIM_CATEGORY_AXIS_LABEL = {
  interval: 0 as const,
  rotate: 0,
  fontSize: DETAIL_CHART_AXIS_FONT_SIZE,
  margin: 10,
  color: DETAIL_AXIS_TEXT_COLOR,
  overflow: 'truncate' as const,
  width: 80,
} as const

/** 详情页类目轴（≤2 列）：不截断，单列名可完整展示 */
const DETAIL_DIM_CATEGORY_AXIS_LABEL_SPARSE = {
  interval: 0 as const,
  rotate: 0,
  fontSize: DETAIL_CHART_AXIS_FONT_SIZE,
  margin: 10,
  color: DETAIL_AXIS_TEXT_COLOR,
} as const

function detailCategoryAxisLabel(catCount: number) {
  return catCount <= 2 ? DETAIL_DIM_CATEGORY_AXIS_LABEL_SPARSE : DETAIL_DIM_CATEGORY_AXIS_LABEL
}

const OP_DETAIL_TWIN_GRID_BOTTOM = DETAIL_DIM_BAR_GRID_BOTTOM

/** 推理 / 训练 / 通信 / 访存详情双柱图：统一上边距（含双系列图例），左右绘图区上下对齐 */
export const DETAIL_DIM_TWIN_BAR_GRID_TOP = 56

/** 详情页数值轴：单位文案在 Y 轴顶端横排（对比页仍用轴左侧竖排 middle + rotate） */
const DETAIL_VALUE_AXIS_NAME = {
  nameLocation: 'end' as const,
  nameRotate: 0,
  nameGap: 12,
  nameTextStyle: { fontSize: DETAIL_CHART_AXIS_FONT_SIZE, color: DETAIL_AXIS_TEXT_COLOR },
  axisLabel: { fontSize: DETAIL_CHART_AXIS_FONT_SIZE, color: DETAIL_AXIS_TEXT_COLOR },
}

export type OpDetailTwinOpts = { gridBottom?: number }

/** 算子详情双图：左右固定同一 grid.bottom；X 轴类目 ≤2 时不截断 */
export function opDetailTwinGridBottom(_shapeLabels: string[], _operatorKeys: string[]): number {
  return OP_DETAIL_TWIN_GRID_BOTTOM
}

/** 类目轴首尾留白：少类目时略加大，避免柱与 y 轴重叠、右侧贴边 */
function detailBarBoundaryGap(catCount: number): boolean | [string, string] {
  if (catCount <= 1) return ['14%', '14%']
  if (catCount <= 3) return ['10%', '10%']
  if (catCount <= 6) return ['6%', '6%']
  return true
}

function maxCategoryLabelLen(categories: string[]): number {
  return categories.reduce((m, s) => Math.max(m, String(s).length), 0)
}

/** 类目轴能横排展示时不旋转（如仅 1 条 megatron-…）；否则再按密度斜排（仅对比页柱图使用） */
function categoryBarLabelsFitHorizontal(categories: string[]): boolean {
  const n = categories.length
  if (n <= 0) return true
  if (n === 1) return true
  const maxLen = maxCategoryLabelLen(categories)
  if (n <= 2 && maxLen <= 28) return true
  if (n <= 4 && maxLen <= 14) return true
  if (n <= 6 && maxLen <= 10) return true
  if (n <= 8 && maxLen <= 8) return true
  return false
}

/** 对比页柱图类目轴：能放下则横排，否则斜排并加大底部留白 */
function categoryBarXAxisUi(categories: string[]): {
  gridBottom: number
  axisLabel: {
    interval: 0
    rotate: number
    fontSize: number
    margin: number
    color: string
  }
} {
  const n = categories.length
  const c = DETAIL_AXIS_TEXT_COLOR
  const fs = DETAIL_CHART_AXIS_FONT_SIZE
  if (categoryBarLabelsFitHorizontal(categories)) {
    return {
      gridBottom: 44,
      axisLabel: { interval: 0, rotate: 0, fontSize: fs, margin: 8, color: c },
    }
  }
  if (n <= 8) {
    return {
      gridBottom: 58,
      axisLabel: { interval: 0, rotate: 28, fontSize: fs, margin: 8, color: c },
    }
  }
  if (n <= 15) {
    return {
      gridBottom: 78,
      axisLabel: { interval: 0, rotate: 42, fontSize: fs, margin: 10, color: c },
    }
  }
  return {
    gridBottom: 102,
    axisLabel: { interval: 0, rotate: 55, fontSize: fs, margin: 10, color: c },
  }
}

/** 详情页推理 / 训练 / 通信 / 访存等柱图：类目横排；≤2 列不截断，多列时省略；底边距统一 */
export function compactDetailBarXAxisUi(categories: string[]): {
  gridBottom: number
  axisLabel:
    | typeof DETAIL_DIM_CATEGORY_AXIS_LABEL
    | typeof DETAIL_DIM_CATEGORY_AXIS_LABEL_SPARSE
} {
  const n = categories.length
  return {
    gridBottom: DETAIL_DIM_BAR_GRID_BOTTOM,
    axisLabel: detailCategoryAxisLabel(n),
  }
}

/** 详情双柱图：左右共用同一 grid.bottom，与类目轴配置一致 */
export function maxCompactDetailBarGridBottom(...categoryLists: (string[] | undefined)[]): number {
  for (const c of categoryLists) {
    if (c?.length) return DETAIL_DIM_BAR_GRID_BOTTOM
  }
  return 0
}

export type DetailDimBarChartOpts = {
  gridTop?: number
  gridBottom?: number
  /** 访存「多型号 + 单模式」柱图：本机系列图例用平台名（如寒武纪） */
  bwPlatBarName?: string
  /** 通信「相对 NVIDIA（%）」双柱图：本机系列图例用平台名 */
  commPlatBarName?: string
  /** 训练「相对 NVIDIA（%）」双柱图：本机系列图例用平台名 */
  trainPlatBarName?: string
  /**
   * 详情平台为 NVIDIA（国际标杆）时：无跨平台对照意义，不画第二根「参考 / 基线」柱，
   * 避免与第一根数值相同却占双簇宽。
   */
  omitTwinBaselineSeries?: boolean
}

export function buildCiLineOption(seriesData: number[]) {
  const cats = [...CI_CHART_LABELS]
  return {
    tooltip: DETAIL_AXIS_TOOLTIP,
    grid: { ...DETAIL_BAR_GRID, top: 28, bottom: 14 },
    xAxis: {
      type: 'category' as const,
      data: cats,
      boundaryGap: detailBarBoundaryGap(cats.length),
      axisLabel: {
        fontSize: DETAIL_CHART_AXIS_FONT_SIZE,
        margin: 10,
        color: DETAIL_AXIS_TEXT_COLOR,
      },
    },
    yAxis: {
      type: 'value' as const,
      min: 0,
      name: '运行次数',
      ...DETAIL_VALUE_AXIS_NAME,
    },
    series: [
      {
        type: 'line' as const,
        name: '运行次数',
        data: seriesData,
        smooth: true,
        symbolSize: 3,
        lineStyle: { color: DETAIL_CHART_PRIMARY, width: 1 },
      },
    ],
  }
}

type OpRow = { shape: string; dtype: string; ic: number; pt: number; scoreEligible?: boolean }

export function buildOpLineOption(rows: OpRow[], opts?: OpDetailTwinOpts) {
  const icSeries = rows.map((r) => {
    const ic = r.ic
    return Number.isFinite(ic) && ic > 0 ? ic : null
  })
  const ptSeries = rows.map((r) => {
    const pt = r.pt
    return Number.isFinite(pt) && pt > 0 ? pt : null
  })
  const shapes = rows.map((r) => r.shape)
  const gridBottom = opts?.gridBottom ?? OP_DETAIL_TWIN_GRID_BOTTOM
  const series = [
    {
      type: 'line' as const,
      name: 'InfiniCore',
      data: icSeries,
      smooth: true,
      symbolSize: 3,
      connectNulls: false,
      lineStyle: { color: DETAIL_CHART_PRIMARY, width: 1 },
    },
    {
      type: 'line' as const,
      name: 'PyTorch',
      data: ptSeries,
      smooth: true,
      symbolSize: 3,
      connectNulls: false,
      lineStyle: { color: DETAIL_CHART_SECONDARY, width: 1 },
    },
  ]
  return {
    tooltip: DETAIL_AXIS_TOOLTIP,
    legend: detailChartLegend(series),
    grid: { ...DETAIL_BAR_GRID, top: OP_DETAIL_TWIN_GRID_TOP, bottom: gridBottom },
    xAxis: {
      type: 'category' as const,
      data: shapes,
      boundaryGap: detailBarBoundaryGap(shapes.length),
      axisLabel: detailCategoryAxisLabel(shapes.length),
    },
    yAxis: {
      type: 'value' as const,
      name: '延迟 (ms)',
      min: 0,
      ...DETAIL_VALUE_AXIS_NAME,
    },
    series,
  }
}

export function buildOpBarAvgOption(opKeys: string[], scores: number[], opts?: OpDetailTwinOpts) {
  const gridBottom = opts?.gridBottom ?? OP_DETAIL_TWIN_GRID_BOTTOM
  return {
    tooltip: DETAIL_AXIS_TOOLTIP,
    grid: {
      ...DETAIL_BAR_GRID,
      top: OP_DETAIL_TWIN_GRID_TOP,
      bottom: gridBottom,
    },
    xAxis: {
      type: 'category' as const,
      data: opKeys,
      axisLabel: detailCategoryAxisLabel(opKeys.length),
      boundaryGap: detailBarBoundaryGap(opKeys.length),
    },
    yAxis: {
      type: 'value' as const,
      name: '得分',
      min: 0,
      ...DETAIL_VALUE_AXIS_NAME,
    },
    series: [
      {
        type: 'bar' as const,
        name: '平均得分',
        data: scores,
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        barCategoryGap: '42%',
        itemStyle: { color: DETAIL_CHART_PRIMARY_SOFT, borderRadius: [2, 2, 0, 0] },
      },
    ],
  }
}

type InferRow = { batch: number; inLen: number; tps: number; ttft?: number }

export function buildInferPrefillBarOption(prefillRows: InferRow[], nvPrefill: InferRow[]) {
  const cats = prefillRows.map((r) => `bs${r.batch} in${r.inLen}`)
  const xUi = compactDetailBarXAxisUi(cats)
  return {
    tooltip: DETAIL_AXIS_TOOLTIP,
    legend: DETAIL_CHART_LEGEND,
    grid: { ...DETAIL_BAR_GRID, top: 56, bottom: xUi.gridBottom },
    xAxis: {
      type: 'category' as const,
      data: cats,
      axisLabel: xUi.axisLabel,
      boundaryGap: detailBarBoundaryGap(cats.length),
    },
    yAxis: { type: 'value' as const, name: 'tokens/s', ...DETAIL_VALUE_AXIS_NAME },
    series: [
      {
        type: 'bar' as const,
        name: 'Prefill TPS',
        data: prefillRows.map((r) => r.tps),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: DETAIL_CHART_PRIMARY_SOFT, borderRadius: [2, 2, 0, 0] },
      },
      {
        type: 'bar' as const,
        name: 'NVIDIA Prefill',
        data: nvPrefill.map((r) => r.tps),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: DETAIL_CHART_SECONDARY_SOFT, borderRadius: [2, 2, 0, 0] },
      },
    ],
  }
}

export function buildInferDecodeBarOption(decodeRows: InferRow[], nvDecode: InferRow[]) {
  const cats = decodeRows.map((r) => `bs${r.batch} in${r.inLen}`)
  const xUi = compactDetailBarXAxisUi(cats)
  return {
    tooltip: DETAIL_AXIS_TOOLTIP,
    legend: DETAIL_CHART_LEGEND,
    grid: { ...DETAIL_BAR_GRID, top: 56, bottom: xUi.gridBottom },
    xAxis: {
      type: 'category' as const,
      data: cats,
      axisLabel: xUi.axisLabel,
      boundaryGap: detailBarBoundaryGap(cats.length),
    },
    yAxis: { type: 'value' as const, name: 'tokens/s', ...DETAIL_VALUE_AXIS_NAME },
    series: [
      {
        type: 'bar' as const,
        name: 'Decode TPS',
        data: decodeRows.map((r) => r.tps),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: DETAIL_CHART_PRIMARY_SOFT, borderRadius: [2, 2, 0, 0] },
      },
      {
        type: 'bar' as const,
        name: 'NVIDIA Decode',
        data: nvDecode.map((r) => r.tps),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: DETAIL_CHART_SECONDARY_SOFT, borderRadius: [2, 2, 0, 0] },
      },
    ],
  }
}

export function buildInferPrefillBarAligned(
  categories: string[],
  platVals: number[],
  nvVals: number[],
  platSeriesName = '当前平台',
  nvSeriesName = 'NVIDIA',
  opts?: DetailDimBarChartOpts,
) {
  const xUi = compactDetailBarXAxisUi(categories)
  const gridTop = opts?.gridTop ?? DETAIL_DIM_TWIN_BAR_GRID_TOP
  const gridBottom = opts?.gridBottom ?? xUi.gridBottom
  const series = [
    {
      type: 'bar' as const,
      name: platSeriesName,
      data: platVals,
      barMaxWidth: DETAIL_BAR_MAX_WIDTH,
      itemStyle: { color: DETAIL_CHART_PRIMARY_SOFT, borderRadius: [2, 2, 0, 0] },
    },
    ...(opts?.omitTwinBaselineSeries
      ? []
      : [
          {
            type: 'bar' as const,
            name: nvSeriesName,
            data: nvVals,
            barMaxWidth: DETAIL_BAR_MAX_WIDTH,
            itemStyle: { color: DETAIL_CHART_SECONDARY_SOFT, borderRadius: [2, 2, 0, 0] },
          },
        ]),
  ]
  return {
    tooltip: DETAIL_AXIS_TOOLTIP,
    legend: detailChartLegend(series),
    grid: { ...DETAIL_BAR_GRID, top: gridTop, bottom: gridBottom },
    xAxis: {
      type: 'category' as const,
      data: categories,
      axisLabel: xUi.axisLabel,
      boundaryGap: detailBarBoundaryGap(categories.length),
    },
    yAxis: { type: 'value' as const, name: 'tokens/s', ...DETAIL_VALUE_AXIS_NAME },
    series,
  }
}

export function buildInferDecodeBarAligned(
  categories: string[],
  platVals: number[],
  nvVals: number[],
  platSeriesName = '当前平台',
  nvSeriesName = 'NVIDIA',
  opts?: DetailDimBarChartOpts,
) {
  const xUi = compactDetailBarXAxisUi(categories)
  const gridTop = opts?.gridTop ?? DETAIL_DIM_TWIN_BAR_GRID_TOP
  const gridBottom = opts?.gridBottom ?? xUi.gridBottom
  const series = [
    {
      type: 'bar' as const,
      name: platSeriesName,
      data: platVals,
      barMaxWidth: DETAIL_BAR_MAX_WIDTH,
      itemStyle: { color: DETAIL_CHART_PRIMARY_SOFT, borderRadius: [2, 2, 0, 0] },
    },
    ...(opts?.omitTwinBaselineSeries
      ? []
      : [
          {
            type: 'bar' as const,
            name: nvSeriesName,
            data: nvVals,
            barMaxWidth: DETAIL_BAR_MAX_WIDTH,
            itemStyle: { color: DETAIL_CHART_SECONDARY_SOFT, borderRadius: [2, 2, 0, 0] },
          },
        ]),
  ]
  return {
    tooltip: DETAIL_AXIS_TOOLTIP,
    legend: detailChartLegend(series),
    grid: { ...DETAIL_BAR_GRID, top: gridTop, bottom: gridBottom },
    xAxis: {
      type: 'category' as const,
      data: categories,
      axisLabel: xUi.axisLabel,
      boundaryGap: detailBarBoundaryGap(categories.length),
    },
    yAxis: { type: 'value' as const, name: 'tokens/s', ...DETAIL_VALUE_AXIS_NAME },
    series,
  }
}

type TrainRow = {
  framework: string
  model: string
  parallel: string
  dtype: string
  flashAttn: string
  tps: number
  baseline: number
  vsA100: number
  note: string
}

export function buildTrainBarThroughput(rows: TrainRow[], opts?: DetailDimBarChartOpts) {
  const categories = rows.map((r) => `${r.framework}·${r.model}`)
  const xUi = compactDetailBarXAxisUi(categories)
  const gridTop = opts?.gridTop ?? DETAIL_DIM_TWIN_BAR_GRID_TOP
  const gridBottom = opts?.gridBottom ?? xUi.gridBottom
  const n = categories.length
  const boundaryGap = detailBarBoundaryGap(n)
  const barCategoryGap = n <= 3 ? '14%' : '36%'
  return {
    tooltip: DETAIL_AXIS_TOOLTIP,
    legend: DETAIL_CHART_LEGEND,
    grid: { ...DETAIL_BAR_GRID, top: gridTop, bottom: gridBottom },
    xAxis: {
      type: 'category' as const,
      data: categories,
      axisLabel: xUi.axisLabel,
      boundaryGap,
    },
    yAxis: { type: 'value' as const, name: 'tokens/process/s', ...DETAIL_VALUE_AXIS_NAME },
    series: [
      {
        type: 'bar' as const,
        name: '实测吞吐 tpps',
        data: rows.map((r) => r.tps),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        barCategoryGap,
        itemStyle: { color: DETAIL_CHART_PRIMARY_SOFT, borderRadius: [2, 2, 0, 0] },
      },
      ...(opts?.omitTwinBaselineSeries
        ? []
        : [
            {
              type: 'bar' as const,
              name: 'NVIDIA 基线',
              data: rows.map((r) => r.baseline),
              barMaxWidth: DETAIL_BAR_MAX_WIDTH,
              barCategoryGap,
              itemStyle: { color: DETAIL_CHART_SECONDARY_SOFT, borderRadius: [2, 2, 0, 0] },
            },
          ]),
    ],
  }
}

export function buildTrainBarVs(rows: TrainRow[], opts?: DetailDimBarChartOpts) {
  const categories = rows.map((r) => `${r.framework}·${r.model}`)
  const xUi = compactDetailBarXAxisUi(categories)
  const gridTop = opts?.gridTop ?? DETAIL_DIM_TWIN_BAR_GRID_TOP
  const gridBottom = opts?.gridBottom ?? xUi.gridBottom
  const platName = String(opts?.trainPlatBarName ?? '').trim() || '本机'
  return {
    tooltip: DETAIL_AXIS_TOOLTIP,
    legend: DETAIL_CHART_LEGEND,
    grid: { ...DETAIL_BAR_GRID, top: gridTop, bottom: gridBottom },
    xAxis: {
      type: 'category' as const,
      data: categories,
      axisLabel: xUi.axisLabel,
      boundaryGap: detailBarBoundaryGap(categories.length),
    },
    yAxis: { type: 'value' as const, name: '% vs NVIDIA', min: 0, ...DETAIL_VALUE_AXIS_NAME },
    series: [
      {
        type: 'bar' as const,
        name: `${platName}（相对 NVIDIA）`,
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: DETAIL_CHART_PRIMARY_SOFT, borderRadius: [2, 2, 0, 0] },
        data: rows.map((r) => r.vsA100),
      },
      ...(opts?.omitTwinBaselineSeries
        ? []
        : [
            {
              type: 'bar' as const,
              name: 'NVIDIA 参考（100%）',
              barMaxWidth: DETAIL_BAR_MAX_WIDTH,
              itemStyle: { color: DETAIL_CHART_SECONDARY_SOFT, borderRadius: [2, 2, 0, 0] },
              data: rows.map(() => 100),
            },
          ]),
    ],
  }
}

type CommRow = {
  linkType: string
  commType: string
  nGpu: number
  bw: number
  baseline: number
  vsA100: number
  note: string
}

export function buildCommBarBw(rows: CommRow[], opts?: DetailDimBarChartOpts) {
  const categories = rows.map((r) => r.commType + ' ' + r.nGpu + 'GPU')
  const xUi = compactDetailBarXAxisUi(categories)
  const gridTop = opts?.gridTop ?? DETAIL_DIM_TWIN_BAR_GRID_TOP
  const gridBottom = opts?.gridBottom ?? xUi.gridBottom
  return {
    tooltip: DETAIL_AXIS_TOOLTIP,
    legend: DETAIL_CHART_LEGEND,
    grid: { ...DETAIL_BAR_GRID, top: gridTop, bottom: gridBottom },
    xAxis: {
      type: 'category' as const,
      data: categories,
      axisLabel: xUi.axisLabel,
      boundaryGap: detailBarBoundaryGap(categories.length),
    },
    yAxis: { type: 'value' as const, name: 'GB/s', ...DETAIL_VALUE_AXIS_NAME },
    series: [
      {
        type: 'bar' as const,
        name: '带宽 GB/s',
        data: rows.map((r) => r.bw),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: DETAIL_CHART_PRIMARY_SOFT, borderRadius: [2, 2, 0, 0] },
      },
      ...(opts?.omitTwinBaselineSeries
        ? []
        : [
            {
              type: 'bar' as const,
              name: 'NVIDIA 基线',
              data: rows.map((r) => r.baseline),
              barMaxWidth: DETAIL_BAR_MAX_WIDTH,
              itemStyle: { color: DETAIL_CHART_SECONDARY_SOFT, borderRadius: [2, 2, 0, 0] },
            },
          ]),
    ],
  }
}

export function buildCommBarVs(rows: CommRow[], opts?: DetailDimBarChartOpts) {
  const categories = rows.map((r) => r.commType)
  const xUi = compactDetailBarXAxisUi(categories)
  const gridTop = opts?.gridTop ?? DETAIL_DIM_TWIN_BAR_GRID_TOP
  const gridBottom = opts?.gridBottom ?? xUi.gridBottom
  const platName = String(opts?.commPlatBarName ?? '').trim() || '本机'
  return {
    tooltip: DETAIL_AXIS_TOOLTIP,
    legend: DETAIL_CHART_LEGEND,
    grid: { ...DETAIL_BAR_GRID, top: gridTop, bottom: gridBottom },
    xAxis: {
      type: 'category' as const,
      data: categories,
      axisLabel: xUi.axisLabel,
      boundaryGap: detailBarBoundaryGap(categories.length),
    },
    yAxis: { type: 'value' as const, name: '% vs NVIDIA', min: 0, ...DETAIL_VALUE_AXIS_NAME },
    series: [
      {
        type: 'bar' as const,
        name: `${platName}（相对 NVIDIA）`,
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: DETAIL_CHART_PRIMARY_SOFT, borderRadius: [2, 2, 0, 0] },
        data: rows.map((r) => r.vsA100),
      },
      ...(opts?.omitTwinBaselineSeries
        ? []
        : [
            {
              type: 'bar' as const,
              name: 'NVIDIA 参考（100%）',
              barMaxWidth: DETAIL_BAR_MAX_WIDTH,
              itemStyle: { color: DETAIL_CHART_SECONDARY_SOFT, borderRadius: [2, 2, 0, 0] },
              data: rows.map(() => 100),
            },
          ]),
    ],
  }
}

type BwRow = {
  model: string
  add: number | null
  copy: number | null
  scale: number | null
  triad: number | null
  avg: number | null
}

export function buildBwBarAvg(
  rows: BwRow[],
  nvidiaAvg: number,
  platSeriesName = '均值 GB/s',
  opts?: DetailDimBarChartOpts,
) {
  const valid = rows.filter((r) => r.avg != null)
  const categories = valid.map((r) => r.model)
  const xUi = compactDetailBarXAxisUi(categories)
  const gridTop = opts?.gridTop ?? DETAIL_DIM_TWIN_BAR_GRID_TOP
  const gridBottom = opts?.gridBottom ?? xUi.gridBottom
  return {
    tooltip: DETAIL_AXIS_TOOLTIP,
    legend: DETAIL_CHART_LEGEND,
    grid: { ...DETAIL_BAR_GRID, top: gridTop, bottom: gridBottom },
    xAxis: {
      type: 'category' as const,
      data: categories,
      axisLabel: xUi.axisLabel,
      boundaryGap: detailBarBoundaryGap(categories.length),
    },
    yAxis: { type: 'value' as const, name: 'GB/s', ...DETAIL_VALUE_AXIS_NAME },
    series: [
      {
        type: 'bar' as const,
        name: platSeriesName,
        data: valid.map((r) => r.avg as number),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: DETAIL_CHART_PRIMARY_SOFT, borderRadius: [2, 2, 0, 0] },
      },
      ...(opts?.omitTwinBaselineSeries
        ? []
        : [
            {
              type: 'bar' as const,
              name: 'NVIDIA 基线',
              data: valid.map(() => nvidiaAvg),
              barMaxWidth: DETAIL_BAR_MAX_WIDTH,
              itemStyle: { color: DETAIL_CHART_SECONDARY_SOFT, borderRadius: [2, 2, 0, 0] },
            },
          ]),
    ],
  }
}

type BwBarMode = 'add' | 'copy' | 'scale' | 'triad'

function bwNvDenomGbps(nvidiaRow: BwRow, m: BwBarMode): number {
  const v = nvidiaRow[m]
  return v != null && Number.isFinite(v) && v > 0 ? (v as number) : BW_NVIDIA_BASELINE_GBPS
}

function bwPctVsNv(platGbps: number | null | undefined, nvDenomGbps: number): number {
  if (platGbps == null || !Number.isFinite(platGbps) || nvDenomGbps <= 0) return 0
  return Math.round((platGbps / nvDenomGbps) * 100)
}

function bwBarPctAxisTooltipFormatter(params: unknown): string {
  if (!Array.isArray(params) || !params.length) return ''
  const first = params[0] as { axisValueLabel?: string; axisValue?: string }
  const title = String(first.axisValueLabel ?? first.axisValue ?? '')
  const lines = (params as { marker?: string; seriesName?: string; value?: number }[])
    .filter((p) => p.seriesName != null)
    .map((p) => `${p.marker ?? ''}${p.seriesName}: ${p.value ?? '—'}%`)
  return title ? `${title}<br/>${lines.join('<br/>')}` : lines.join('<br/>')
}

const BW_BAR_MODES_PCT_TOOLTIP = {
  trigger: 'axis' as const,
  textStyle: { fontSize: DETAIL_CHART_AXIS_FONT_SIZE, color: DETAIL_AXIS_TEXT_COLOR },
  formatter: bwBarPctAxisTooltipFormatter,
}

export function buildBwBarModes(
  platRowOrRows: BwRow | BwRow[],
  nvidiaRow: BwRow,
  singleMode?: BwBarMode | null,
  opts?: DetailDimBarChartOpts,
) {
  const platRows = Array.isArray(platRowOrRows) ? platRowOrRows : [platRowOrRows]
  const gridTop = opts?.gridTop ?? DETAIL_DIM_TWIN_BAR_GRID_TOP

  /** 顶栏指定单一模式时：X 轴为各型号；数值为相对 NVIDIA 同模式带宽的百分比，参考柱恒为 100% */
  if (singleMode) {
    const mk = singleMode
    const valid = platRows.filter((r) => r[mk] != null && Number.isFinite(r[mk] as number))
    if (!valid.length) {
      return {}
    }
    const categories = valid.map((r) => r.model || '—')
    const xUi = compactDetailBarXAxisUi(categories)
    const gridBottom = opts?.gridBottom ?? xUi.gridBottom
    const nvDenom = bwNvDenomGbps(nvidiaRow, mk)
    const platSeriesName =
      valid.length > 1
        ? `${opts?.bwPlatBarName ?? valid[0]?.model ?? '本机'}（${mk}）`
        : `${valid[0]?.model || '自研'}（${mk}）`
    const nvSeriesName = 'NVIDIA 参考（100%）'
    return {
      tooltip: BW_BAR_MODES_PCT_TOOLTIP,
      legend: DETAIL_CHART_LEGEND,
      grid: { ...DETAIL_BAR_GRID, top: gridTop, bottom: gridBottom },
      xAxis: {
        type: 'category' as const,
        data: categories,
        axisLabel: xUi.axisLabel,
        boundaryGap: detailBarBoundaryGap(categories.length),
      },
      yAxis: { type: 'value' as const, name: '% vs NVIDIA', min: 0, ...DETAIL_VALUE_AXIS_NAME },
      series: [
        {
          type: 'bar' as const,
          name: platSeriesName,
          data: valid.map((r) => bwPctVsNv(r[mk] as number, nvDenom)),
          barMaxWidth: DETAIL_BAR_MAX_WIDTH,
          itemStyle: { color: DETAIL_CHART_PRIMARY_SOFT, borderRadius: [2, 2, 0, 0] },
        },
        ...(opts?.omitTwinBaselineSeries
          ? []
          : [
              {
                type: 'bar' as const,
                name: nvSeriesName,
                data: valid.map(() => 100),
                barMaxWidth: DETAIL_BAR_MAX_WIDTH,
                itemStyle: { color: DETAIL_CHART_SECONDARY_SOFT, borderRadius: [2, 2, 0, 0] },
              },
            ]),
      ],
    }
  }

  const bestRow = platRows[0]
  const modes = ['add', 'copy', 'scale', 'triad'] as const
  const categories = [...modes]
  const xUi = compactDetailBarXAxisUi(categories)
  const gridBottom = opts?.gridBottom ?? xUi.gridBottom
  const platSeriesName = bestRow.model || '自研'
  const nvSeriesName = 'NVIDIA 参考（100%）'
  return {
    tooltip: BW_BAR_MODES_PCT_TOOLTIP,
    legend: DETAIL_CHART_LEGEND,
    grid: { ...DETAIL_BAR_GRID, top: gridTop, bottom: gridBottom },
    xAxis: {
      type: 'category' as const,
      data: categories,
      axisLabel: xUi.axisLabel,
      boundaryGap: detailBarBoundaryGap(categories.length),
    },
    yAxis: { type: 'value' as const, name: '% vs NVIDIA', min: 0, ...DETAIL_VALUE_AXIS_NAME },
    series: [
      {
        type: 'bar' as const,
        name: platSeriesName,
        data: modes.map((m) => bwPctVsNv(bestRow[m], bwNvDenomGbps(nvidiaRow, m))),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: DETAIL_CHART_PRIMARY_SOFT, borderRadius: [2, 2, 0, 0] },
      },
      ...(opts?.omitTwinBaselineSeries
        ? []
        : [
            {
              type: 'bar' as const,
              name: nvSeriesName,
              data: modes.map(() => 100),
              barMaxWidth: DETAIL_BAR_MAX_WIDTH,
              itemStyle: { color: DETAIL_CHART_SECONDARY_SOFT, borderRadius: [2, 2, 0, 0] },
            },
          ]),
    ],
  }
}

type CardLite = {
  key: string
  ownScore: number | null
  ownVal?: string | null
}

type PlatLite = { key: string; name: string; color: string }

/** 算子：自研相对开源的提速倍率；其余维度：各平台相对 NVIDIA=100 的得分 */
export type CompareScoreBarMode = 'operator' | 'platformScore'

export function buildCompareScoreBar(
  cards: CardLite[],
  plats: PlatLite[],
  mode: CompareScoreBarMode = 'operator',
) {
  const categories = plats.map((p) => p.name)
  const xUi = categoryBarXAxisUi(categories)
  const gridBottom = categoryBarLabelsFitHorizontal(categories)
    ? DETAIL_DIM_BAR_GRID_BOTTOM
    : xUi.gridBottom
  const isOperator = mode === 'operator'
  return {
    tooltip: {
      trigger: 'axis' as const,
      textStyle: { fontSize: DETAIL_CHART_AXIS_FONT_SIZE, color: DETAIL_AXIS_TEXT_COLOR },
    },
    legend: { ...DETAIL_CHART_LEGEND, right: 8 },
    grid: { ...DETAIL_BAR_GRID, left: COMPARE_BAR_GRID_LEFT, top: 44, bottom: gridBottom },
    xAxis: {
      type: 'category' as const,
      data: categories,
      axisLabel: xUi.axisLabel,
      boundaryGap: detailBarBoundaryGap(categories.length),
    },
    yAxis: {
      type: 'value' as const,
      name: isOperator ? '提速倍率（×）' : '相对NVIDIA得分',
      nameLocation: 'middle' as const,
      nameRotate: 90,
      nameGap: COMPARE_Y_NAME_GAP_SCORE,
      nameTextStyle: {
        fontSize: DETAIL_CHART_AXIS_FONT_SIZE,
        color: DETAIL_AXIS_TEXT_COLOR,
      },
      min: 0,
      axisLabel: {
        fontSize: DETAIL_CHART_AXIS_FONT_SIZE,
        formatter: isOperator ? (v: number) => v + '×' : undefined,
        color: DETAIL_AXIS_TEXT_COLOR,
      },
      splitLine: { lineStyle: { color: '#f0f0f0' } },
    },
    series: [
      {
        type: 'bar' as const,
        name: isOperator ? '自研提速倍率（×）' : '平台得分',
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: DETAIL_CHART_PRIMARY_SOFT, borderRadius: [2, 2, 0, 0] },
        data: cards.map((c) =>
          isOperator
            ? c.ownScore != null
              ? Number((c.ownScore / 100).toFixed(2))
              : 0
            : (c.ownScore ?? 0),
        ),
      },
      {
        type: 'bar' as const,
        name: isOperator ? '开源基准（×1.0）' : 'NVIDIA基线（100）',
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        /** 与详情/概览双柱辅色一致（绿），图例与柱同色 */
        itemStyle: { color: DETAIL_CHART_SECONDARY_SOFT, borderRadius: [2, 2, 0, 0] },
        data: cards.map(() => (isOperator ? 1 : 100)),
      },
    ],
  }
}

/** 各平台代表延迟同一指标，柱色与详情/概览柱状图主系列一致（不用平台品牌色） */
export function buildCompareLatencyBar(names: string[], latencies: number[]) {
  const xUi = categoryBarXAxisUi(names)
  const gridBottom = categoryBarLabelsFitHorizontal(names) ? DETAIL_DIM_BAR_GRID_BOTTOM : xUi.gridBottom
  return {
    tooltip: {
      trigger: 'axis' as const,
      textStyle: { fontSize: DETAIL_CHART_AXIS_FONT_SIZE, color: DETAIL_AXIS_TEXT_COLOR },
    },
    grid: { ...DETAIL_BAR_GRID, left: COMPARE_BAR_GRID_LEFT, top: 28, bottom: gridBottom },
    xAxis: {
      type: 'category' as const,
      data: names,
      axisLabel: xUi.axisLabel,
      boundaryGap: detailBarBoundaryGap(names.length),
    },
    yAxis: {
      type: 'value' as const,
      name: 'ms，越低越好',
      nameLocation: 'middle' as const,
      nameRotate: 90,
      nameGap: COMPARE_Y_NAME_GAP_LATENCY,
      nameTextStyle: {
        fontSize: DETAIL_CHART_AXIS_FONT_SIZE,
        color: DETAIL_AXIS_TEXT_COLOR,
      },
      min: 0,
      axisLabel: {
        fontSize: DETAIL_CHART_AXIS_FONT_SIZE,
        color: DETAIL_AXIS_TEXT_COLOR,
      },
      splitLine: { lineStyle: { color: '#f0f0f0' } },
    },
    series: [
      {
        type: 'bar' as const,
        name: '自研延迟 ms',
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        data: latencies.map((v) => ({
          value: v,
          itemStyle: { color: DETAIL_CHART_PRIMARY_SOFT, borderRadius: [2, 2, 0, 0] },
        })),
      },
    ],
  }
}
