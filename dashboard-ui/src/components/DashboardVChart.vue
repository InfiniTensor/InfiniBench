<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import type { ECBasicOption } from 'echarts/types/dist/shared'
import type { SetOptionOpts } from 'echarts/types/dist/shared'
import VChart from 'vue-echarts'

const props = withDefaults(
  defineProps<{
    option: ECBasicOption
    updateOptions?: SetOptionOpts
    /** 与全局 detail 图一致：切换维度时 notMerge，避免轴样式残留 */
    notMerge?: boolean
  }>(),
  { notMerge: false },
)

const chartRef = ref<InstanceType<typeof VChart> | null>(null)

const resolvedUpdateOptions = computed((): SetOptionOpts | undefined => {
  if (props.updateOptions) return props.updateOptions
  if (!props.notMerge) return undefined
  return { notMerge: true, replaceMerge: ['series', 'legend'] }
})

function resize() {
  chartRef.value?.resize()
}

defineExpose({ resize })

watch(
  () => props.option,
  () => {
    if (props.notMerge) {
      chartRef.value?.clear()
    }
    void nextTick(() => {
      requestAnimationFrame(resize)
    })
  },
  { deep: true, flush: 'pre' },
)
</script>

<template>
  <v-chart
    ref="chartRef"
    :option="option"
    :update-options="resolvedUpdateOptions"
    autoresize
    v-bind="$attrs"
  />
</template>
