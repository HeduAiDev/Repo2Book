# ch17《控制流下降到结构化 IR：if/for/while 如何变成 scf 与 φ》定稿

- **Type**: delivery
- **Chapter**: 17
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T02:35:04Z
- **Agents involved**: archivist, team-lead
- **User present**: False
- **Tags**: ch17, control-flow, scf, f13-resolved, f16-planted, bible-archive

## What happened

全书前端最烧脑的皇冠明珠章：补全 ch15 只走通两条路径(visit_For range / visit_if_scf)之外的全部控制流下降台阶。①visit_If 四路分流(动态/静态 cond × 带/不带 return)——ContainsReturnChecker 静态判 return(跨 JITFunction 递归)决定走 visit_if_top_level(顶层 CFG，cf.cond_br + endif 块参数手写 φ)还是 visit_if_scf(结构化，create_if_op + yield)；②visit_Return 建 post_ret_block 死块(TTIR `no predecessors`)实证 return 破坏单入单出；③scf_stack 状态栈禁止循环体内 return；④visit_For 三分流(static_range 编译期整体展开/tl.range 带 num_stages+loop_unroll_factor/裸 range)+负步长翻转(lb/ub 交换,体首 iv=ub-iv+lb 反算)+诱导变量 create_poison 占位后 replace_all_uses_with 回填+num_stages/loop_unroll_factor 追踪期即挂成 tt. 属性；⑤visit_While 的 before/after 双区域。全部经 pin v3.2.0 headless ASTSource.make_ir 精确编译取证。15 机制,10 图全 blind PASS,review APPROVED(5 条 negotiable/non-blocking)。

## Why it matters

回收伏笔 f13(SSA/φ/块参数三层地基,ch15 埋→ch17 补全);新开伏笔 f16(num_stages/loop_unroll_factor 属性挂载→ch30 消费)。是全书控制流下降的完整落点，为 ch25(AxisInfo)/ch29-30(软件流水线 pass)埋下必需的地基。

## What to remember

ch17 已定稿归档：f13 回收(resolved_in=ch17)、f16 新开(plant ch17→payoff ch30)；glossary 194→204(新增10条:visit_If/visit_For/visit_While/cf.cond_br/ContainsReturnChecker/scf_stack/负步长翻转/poison诱导变量/before-after双区域/死块，并更新scf.for/scf.if/num_stages/loop_unroll_factor四条既有词条);concepts.json+6;interfaces.json新增ch17键。一致性核验通过:f4/f5/f7/f11/f12/f13均resolved且payoff==resolved_in;f14→ch20/f15→ch24/f16→ch30仍open,无并行泄漏。
