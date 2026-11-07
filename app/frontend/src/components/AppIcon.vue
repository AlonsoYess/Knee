<script setup lang="ts">
import { computed } from 'vue'
import {
  Search, Plus, ArrowRight, ChevronLeft, ChevronRight, FileText, Image,
  Clock3, Check, CircleHelp, SlidersHorizontal, X, LogOut, ShieldCheck,
  UserRound, ChevronDown, ArrowUpRight, Eye, EyeOff, Mail, LockKeyhole,
  ClipboardList, ScanLine, Activity,
} from '@lucide/vue'

const props = withDefaults(defineProps<{ name: string; size?: number }>(), { size: 20 })
const icons = {
  search: Search, plus: Plus, 'arrow-right': ArrowRight, 'chevron-left': ChevronLeft,
  'chevron-right': ChevronRight, file: FileText, image: Image, clock: Clock3,
  check: Check, help: CircleHelp, filter: SlidersHorizontal, x: X, logout: LogOut,
  shield: ShieldCheck, user: UserRound, 'chevron-down': ChevronDown,
  'arrow-up-right': ArrowUpRight, eye: Eye, 'eye-off': EyeOff, mail: Mail,
  lock: LockKeyhole, evaluations: ClipboardList, scan: ScanLine, activity: Activity,
}
const icon = computed(() => icons[props.name as keyof typeof icons] ?? CircleHelp)
</script>

<template><component :is="icon" :size="size" :stroke-width="1.7" aria-hidden="true" class="app-icon" /></template>
