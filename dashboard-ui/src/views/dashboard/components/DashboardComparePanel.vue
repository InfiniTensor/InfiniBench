<script setup lang="ts">
import { computed, ref, toValue, watch } from 'vue'
import type { ColumnsType } from 'ant-design-vue/es/table'
import DashboardVChart from '@/components/DashboardVChart.vue'
import { PLATFORMS } from '@/data'
import type { CardRow } from '@/features/dashboard/dashboardFilterHelpers'
import { useInfiniDashboard } from '@/composables/useInfiniDashboard'
import { useDashboardNavigation } from '@/composables/useDashboardNavigation'
import { useEchartsGridResize } from '@/composables/useEchartsGridResize'
import { scoreTierColor } from '@/utils/scoreColor'

/** public 根目录下的品牌图（与文件名一致） */
const PLAT_LOGO_FILE: Record<string, string> = {
  nvidia: 'nvidia.png',
  mthreads: 'moore.png',
  cambricon: 'cambricon.png',
  metax: 'metax.png',
  iluvatar: 'iluvatar.png',
  ascend: 'ascend.png',
  hygon: 'hygon.png',
  generic: 'ali.png',
}

function platLogoHref(platKey: string): string {
  const file = PLAT_LOGO_FILE[platKey]
  if (!file) return ''
  const base = import.meta.env.BASE_URL || '/'
  return base.endsWith('/') ? `${base}${file}` : `${base}/${file}`
}

type CompareTableRecord = {
  plat: (typeof PLATFORMS)[number]
  card: CardRow
  delta: number | null
}

function compareRowKey(r: CompareTableRecord) {
  return r.plat.key
}

function compareDeltaText(delta: number | null) {
  if (delta == null) return '—'
  if (delta >= 0) return `+${delta}%`
  return `${delta}%`
}

const {
  currentView,
  activeDimKey,
  comparePageTitle,
  compareKpiBlocks,
  compareNvidiaBaselineCard,
  compareScoreOption,
  compareLatencyOption,
  compareTableRows,
} = useInfiniDashboard()

function compareOpenValText(record: CompareTableRecord): string {
  if (activeDimKey.value === 'op') return record.card.openVal || '—'
  if (record.plat.key === 'nvidia') return '—'
  return compareNvidiaBaselineCard.value?.ownVal || '—'
}

/** 列宽总和与 scroll.x 一致（与详情页数据明细 a-table 用法对齐） */
const compareTableColumns = computed((): ColumnsType<CompareTableRecord> => {
  const isOp = activeDimKey.value === 'op'
  return [
    { title: '平台', key: 'plat', width: 168, ellipsis: true },
    { title: '类型', key: 'platType', width: 96, ellipsis: true },
    { title: '平均得分', key: 'ownScore', width: 104 },
    {
      title: isOp ? '自研代表值' : '代表值',
      key: 'ownVal',
      minWidth: 120,
      width: 132,
      ellipsis: true,
    },
    {
      title: isOp ? '开源参考值' : 'NVIDIA基线值',
      key: 'openVal',
      width: isOp ? 132 : 140,
      ellipsis: true,
    },
    { title: '测试条数', key: 'n', width: 96 },
    { title: '配置', key: 'extra', minWidth: 140, width: 200, ellipsis: true },
    { title: '相对 NVIDIA', key: 'delta', width: 112 },
  ]
})

const compareScoreChartIsOperator = computed(() => activeDimKey.value === 'op')
const compareScoreChartTitle = computed(() =>
  compareScoreChartIsOperator.value ? '自研提速倍率对比' : '性能得分对比',
)
const compareScoreChartDesc = computed(() =>
  compareScoreChartIsOperator.value
    ? '相对开源基准的提速倍率 · 灰线为基准×1.0 · 越高越好'
    : '以NVIDIA=100为基准 · 越高越好',
)
const { goOverview } = useDashboardNavigation()

function hasChart(opt: object) {
  return opt && Object.keys(opt).length > 0
}

const showLatencyChart = computed(
  () => hasChart(compareLatencyOption.value as object),
)

const compareChartsGridRef = ref<HTMLElement | null>(null)
const compareScoreChartRef = ref<InstanceType<typeof DashboardVChart> | null>(null)
const compareLatencyChartRef = ref<InstanceType<typeof DashboardVChart> | null>(null)

const { scheduleResize: scheduleCompareChartsResize } = useEchartsGridResize(
  compareChartsGridRef,
  () => [compareScoreChartRef.value, compareLatencyChartRef.value],
  () => [toValue(compareScoreOption), toValue(compareLatencyOption), showLatencyChart.value],
)

watch(
  currentView,
  (v) => {
    if (v !== 'compare') return
    scheduleCompareChartsResize()
  },
  { flush: 'post' },
)
</script>

<template>
  <div class="compare-panel">
    <header class="compare-panel__header">
      <div class="detail-title-row">
        <div class="detail-title-group">
          <span class="detail-title">{{ comparePageTitle }}</span>
        </div>
        <button type="button" class="detail-back-btn" @click="goOverview">返回概览</button>
      </div>
    </header>
    <div class="compare-panel__scroll">
      <div class="compare-kpi-grid">
        <div
          v-for="row in compareKpiBlocks"
          :key="row.plat.key"
          class="compare-kpi-card"
          :class="{ 'compare-kpi-card--best': row.isBest }"
        >
          <div
            v-if="platLogoHref(row.plat.key)"
            class="compare-kpi-logo-wrap"
            aria-hidden="true"
          >
            <img
              class="compare-kpi-logo"
              :src="platLogoHref(row.plat.key)"
              :alt="`${row.plat.name} 品牌`"
              loading="lazy"
            />
          </div>
          <div class="compare-kpi-body">
            <span class="compare-kpi-head">
              <span class="compare-kpi-brand">{{ row.plat.name }}</span>
              <span
                class="compare-kpi-score"
                :style="{ color: scoreTierColor(row.card.ownScore) }"
              >
                {{ row.card.ownScore ?? '—' }}
              </span>
            </span>
            <span class="compare-kpi-meta">
              {{ row.card.ownVal || '—' }} · {{ row.card.extra || '' }} · 排名 {{ row.rank }}
            </span>
          </div>
          <span v-if="row.isBest" class="compare-kpi-best-badge">最高</span>
        </div>
      </div>

    <div
      ref="compareChartsGridRef"
      class="charts-grid"
      :class="{ 'charts-grid--compare-single-chart': !showLatencyChart }"
    >
      <div class="chart-card">
        <div class="chart-title">{{ compareScoreChartTitle }}</div>
        <div class="compare-chart-desc">{{ compareScoreChartDesc }}</div>
        <DashboardVChart
          v-if="hasChart(compareScoreOption)"
          ref="compareScoreChartRef"
          class="compare-chart"
          :option="compareScoreOption"
        />
      </div>
      <div v-if="showLatencyChart" class="chart-card">
        <div class="chart-title">代表延迟对比（ms，越低越好）</div>
        <div class="compare-chart-desc">
          各平台代表算子延迟值 · 单位 ms · 越低越好
        </div>
        <DashboardVChart
          ref="compareLatencyChartRef"
          class="compare-chart"
          :option="compareLatencyOption"
        />
      </div>
    </div>

    <div class="table-card">
      <div class="table-head-row">
        <div class="table-title">对比明细</div>
        <div class="compare-table-hint">数值来自当前维度卡片数据</div>
      </div>
      <div class="table-wrap">
        <a-table
          v-if="compareTableRows.length"
          :columns="compareTableColumns"
          :data-source="(compareTableRows as CompareTableRecord[])"
          :pagination="false"
          :bordered="false"
          table-layout="fixed"
          :scroll="{ x: 1040 }"
          :row-key="compareRowKey"
        >
          <template #bodyCell="{ column, record }">
            <template v-if="column.key === 'plat'">
              <span
                class="compare-dot"
                style="display: inline-block; margin-right: 8px; vertical-align: middle"
                :style="{ background: record.plat.color }"
              />
              {{ record.plat.name }}
            </template>
            <template v-else-if="column.key === 'platType'">
              {{ record.plat.type }}
            </template>
            <template v-else-if="column.key === 'ownScore'">
              <span
                class="score-num"
                :style="{ color: scoreTierColor(record.card.ownScore), fontWeight: 700 }"
              >
                {{ record.card.ownScore ?? '—' }}
              </span>
            </template>
            <template v-else-if="column.key === 'ownVal'">
              <span class="compare-table-val-plain">{{ record.card.ownVal || '—' }}</span>
            </template>
            <template v-else-if="column.key === 'openVal'">
              <span class="compare-table-val-plain">{{
                compareOpenValText(record as CompareTableRecord)
              }}</span>
            </template>
            <template v-else-if="column.key === 'n'">
              {{ record.card.n ?? '—' }}
            </template>
            <template v-else-if="column.key === 'extra'">
              {{ record.card.extra || '—' }}
            </template>
            <template v-else-if="column.key === 'delta'">
              <span
                v-if="record.delta != null"
                :class="record.delta >= 0 ? 'score-up' : 'score-dn'"
                class="score-num"
              >
                {{ compareDeltaText(record.delta) }}
              </span>
              <span v-else class="score-num score-na">—</span>
            </template>
          </template>
        </a-table>
      </div>
    </div>
    </div>
  </div>
</template>

<style scoped>
.compare-panel {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  height: 100%;
  background-color: #fff;
}
.compare-panel__header {
  flex-shrink: 0;
  margin-bottom: 13px;
}
.compare-kpi-card {
  display: flex;
  align-items: center;
  gap: 14px;
  box-sizing: border-box;
  min-width: 0;
  padding: 18px 20px;
  border-radius: 8px;
  background: rgb(244, 244, 244);
  border: none;
}
.compare-kpi-card--best {
  background: rgb(226 228 238);
}
.compare-kpi-logo-wrap {
  flex-shrink: 0;
  width: 56px;
  height: 56px;
  box-sizing: border-box;
  padding: 6px;
  border-radius: 8px;
  background: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
}
.compare-kpi-logo {
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
  object-fit: contain;
}
.compare-kpi-body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  column-gap: 10px;
  row-gap: 4px;
}
.compare-kpi-head {
  display: inline-flex;
  flex-wrap: nowrap;
  align-items: center;
  gap: 8px 14px;
  flex-shrink: 0;
}
.compare-kpi-brand {
  font-size: 16px;
  font-weight: 700;
  line-height: 1.2;
  color: rgb(13, 21, 60);
}
/* 得分色由 scoreTierColor(ownScore) 内联，与概览卡 ownScoreColStyle 同源 */
.compare-kpi-score {
  font-size: 32px;
  font-weight: 700;
  line-height: 1.2;
}
.compare-kpi-meta {
  flex: 1 1 200px;
  min-width: 0;
  font-size: 14px;
  font-weight: 400;
  line-height: 1.4;
  color: #9a9a9a;
  margin: 0;
}
/* 与「返回概览」.detail-back-btn 配色、高度、圆角、字号一致（非按钮，无 pointer） */
.compare-kpi-best-badge {
  flex-shrink: 0;
  box-sizing: border-box;
  margin: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 28px;
  padding: 0 14px;
  border: none;
  border-radius: 6px;
  background: rgb(125, 134, 173);
  color: #fff;
  font-size: 16px;
  font-weight: 600;
  line-height: 26px;
  font-family: inherit;
  cursor: default;
}
.compare-chart-desc {
  margin: 0 0 8px;
  font-size: 14px;
  line-height: 1.55;
  color: #8c8c8c;
}
.compare-table-hint {
  font-size: 14px;
  line-height: 1.55;
  color: #8c8c8c;
}
.compare-panel__scroll {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  overflow-x: hidden;
  padding-bottom: 4px;
  background-color: #fff;
}
.compare-panel__scroll .compare-kpi-grid {
  margin-bottom: 28px;
}
.compare-panel__scroll :deep(.ant-table-thead > tr > th.ant-table-cell),
.compare-panel__scroll :deep(.ant-table-tbody > tr > td.ant-table-cell) {
  text-align: left !important;
}
.compare-table-val-plain {
  color: #000000;
  font-weight: 400;
}
.compare-panel__scroll .charts-grid {
  gap: 18px;
  margin-bottom: 28px;
}
/** 无代表延迟图时仅一卡，避免仍占半宽 */
.compare-panel__scroll .charts-grid.charts-grid--compare-single-chart {
  grid-template-columns: 1fr;
}
.compare-panel__scroll .table-card {
  margin-bottom: 28px;
  border: none;
  padding: 0;
}
.compare-panel__scroll .table-head-row {
  margin-bottom: 8px;
}
.compare-panel__scroll .chart-card {
  border: none;
  padding: 0;
}
.compare-panel__scroll .table-title {
  font-size: 16px;
  font-weight: 700;
  color: rgb(13, 21, 60);
  line-height: 26px;
}
.compare-panel__scroll .charts-grid .chart-title {
  font-size: 16px;
  font-weight: 700;
  color: rgb(13, 21, 60);
  line-height: 26px;
  margin-bottom: 8px;
}
.compare-panel__scroll .charts-grid .chart-card {
  padding: 0;
}
.compare-chart {
  height: 280px;
  width: 100%;
  display: block;
  line-height: 0;
}
.charts-grid .chart-card {
  padding-bottom: 0;
}
</style>
