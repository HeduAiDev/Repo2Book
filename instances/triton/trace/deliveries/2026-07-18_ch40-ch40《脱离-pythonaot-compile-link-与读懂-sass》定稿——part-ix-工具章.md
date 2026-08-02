# ch40《脱离 Python:AOT compile/link 与读懂 SASS》定稿——Part IX 工具章

- **Type**: delivery
- **Chapter**: ch40
- **Date**: 2026-07-18
- **Timestamp**: 2026-07-18T07:30:00+08:00
- **Agents involved**: archivist
- **User present**: False
- **Tags**: triton, part-9, deep, skip_impl, aot, compile.py, link.py, disasm.py, from_hints, kernel_suffix, HeaderParser, tt-linker, compute_spec_key-c, cuobjdump, sass, parseCtrl, ctrl-word, salvage

## What happened

Part IX 工具章(kind=deep/skip_impl),verdict=APPROVED。主题:把一个 Triton 核推到一生两个端点——脱离 Python 部署、读懂它编成了什么。三工具主线:

(1)`compile.py` AOT 编译。§1 一行带整除性提示的签名(`*fp32:16, i32:16, 1024, i32`)逗号切段三分成 hints(带 :16/:1 后缀→{位号:值})/constants(裸整数 1024)/signature(运行期原型);仅 `:16`(可被16整除,tt.divisibility)与 `:1`(恒等于1,tt.equal_to)两种合法提示。§2 `AttrsDescriptor.from_hints` 拿 property_values 对照表把命令行数字物化成特化,与 JIT 期从参数真实值推断产出同一个 AttrsDescriptor——两条构造路径是 AOT/JIT 在特化上的唯一分岔(回指 ch14);`:1` 参数经 get_constants() 并回 constants、从运行期原型消失。§3 `binascii.hexlify` 把 cubin 逐字节抄进 C:5648 字节→11296 十六进制字符→`unsigned char CUBIN_NAME[11296]`,头 6 字节 `0x7f 45 4c 46 02 01` 即 ELF 魔数(cubin 本质 ELF);配 cuModuleLoadData(内存直读)+cuLaunchKernel 灌进 compile.c/compile.h 模板,烙成脱离 Python 的自包含 C。

(2)`link.py` 多核链接。§4 特化信息写进函数名当暗号:`kernel_suffix`(发报机)逐参数拼位号、恒1追'c'/对齐16追'd'(suffix='012d'),link.py `_match_suffix`(收报机)扫这串解回每参数提示——跨两个内存不共享的命令行程序传特化的隐形信道。§5 `HeaderParser` 三条正则捞回 compile.h 尾部 tt-linker 漂流瓶注释(核名/C签名/algo_info→_match_name→_match_c_sig→_match_suffix),塞进 KernelLinkerMeta 归组。§6 `make_kernel_hints_dispatcher` 生成运行期整除性分派链:先试约束最强(N%16==0)那份、退而求其次、全不中报错——这正是 JIT 期 `compute_spec_key`(ch14)的 AOT C 化身。§7 `make_func_pointers`/algo_id 函数指针表是两级分派第二级:表里每项本身是一条整除性 if 链。

(3)`disasm.py` SASS 反汇编。§8 调 `cuobjdump -sass` 出 SASS,每条指令占两行(FLINE 汇编体+首半64位编码 / SLINE 次半64位控制字),disasm line_idx+=2 折叠成一条。§9(全章最该看懂)`parseCtrl`:Volta 起 ptxas 把调度信息编进每条指令后附的 64 位控制字(硬件记分牌成本高),五次右移+掩码抠出五个互不重叠位段——stall(41-44,发射后停几拍)/yield(45,值0才输出'Y')/wr-barrier(46-48)/rd-barrier(49-51,7=无)/wait-mask(52-57),格式化成左列 `wait:read:write:yield:stall`(例 `02:-:-:Y:d`=等2号屏障、让位、停13拍);读它=读 ptxas 调度决策,stall 普遍大+wait-mask 频繁命中=核在等访存,比看 occupancy 更贴近真相(往 ch37 ptxas→cubin 再下一层看时序)。§10 BRA→LBB 两趟重标 + 惰性反汇编(访问 kernel.asm['sass'] 这个此前不存在的键才 fork cuobjdump+lru_cache)。

诚实边界:compile.py/link.py/disasm.py 真 Python 源逐字可核;host 无 GPU/无 cuobjdump 处 compile.c/h 烙制、cuobjdump -sass 标『需真机』,能跑照跑(§1 签名三分/§4 suffix 编解码/§9 parseCtrl 位解码均纯 Python host 复现)。disasm.py 位定义源自 Da Yan cuobjdump 逆向(MIT)、非 NVIDIA 官方,适用 Volta~Hopper,跨 SM 架构可能变动。

质量事件:2 轮 write-review + blind round 1(PASS)+ map round 1(PASS),impl/test 按 skip_impl 跳过。5 项 issue 全 non-blocking reader-comp,Lead 派 writer 落地补齐(不伤质量):§6 把 compute_spec_key 符号接回 ch14、§3 首现补 sig_hash 是什么、§10 惰性反汇编改写成不抛 AsmDict 类名、§5 KernelLinkerMeta 两同值字段说明、§3 suffix 黑盒预告。workflow 有 1 个并行 verify slot 瞬时 StructuredOutput 重试超限(agent-error)——已 salvage,不影响 review_verdict=APPROVED/Map pass。本章无伏笔埋/回收(bible.py due ch40 确为空,未 resolve 任何伏笔、未动 arc-map.json)。skip_impl 无精简版接口(按契约跳过 interfaces 登记)。bible 回写:glossary +19(490→509)、concepts +16(329→345)、figures +6(83→89,含 chapter-map)。

## Why it matters

本章是 Triton 核的部署与验收终点:AOT compile/link 让核脱离 Python 当普通 C 库交付到生产(把 ch14 JIT 特化钉到命令行),读 SASS 控制字给出 profile 慢核的硬尺——stall 普遍大、wait-mask 频繁命中即核在等访存,该去优化合并访存/加 num_stages/换 tile,而非盲堆 occupancy。它把 ch37「ptxas 怎么把 PTX 编成 cubin」延续成「怎么亲眼验收它排得好不好」,收束 Part IX 工具章的部署+度量闭环。过程教训:并行 verify slot 瞬时 agent-error 不该连累已 PASS 的 review/map 判定——Lead salvage 是对的;reader-comp non-blocking issue 由 writer 定点补钩子、不退整章,与 ch39 review-exhausted 后的落地方式一致。

## What to remember

Part IX 工具章(deep/skip_impl,APPROVED)。三工具:compile.py AOT(签名三分 hints/constants/signature→from_hints 物化特化=JIT 同一 AttrsDescriptor 的命令行入口→hexlify 烙 cubin 进 compile.c/h,ELF 魔数打头,脱离 Python 自包含 C);link.py 链接(kernel_suffix↔_match_suffix 函数名当跨进程暗号→HeaderParser 捞 tt-linker 漂流瓶→make_kernel_hints_dispatcher 运行期整除性 if 链=compute_spec_key 的 C 化身→algo_id 函数指针表是两级分派第二级);disasm.py 反汇编(cuobjdump -sass 两行一指令折叠→parseCtrl 解 64 位控制字五位段 stall/yield/wr-barrier/rd-barrier/wait-mask=ptxas 调度决策→BRA→LBB 重标+惰性反汇编)。回指 ch14 特化、ch37 ptxas/cubin。无伏笔;skip_impl 无接口;host 无 GPU 处标需真机、能跑照跑;disasm 位定义 Da Yan 逆向(Volta~Hopper)。salvage:并行 verify slot 瞬时 StructuredOutput error 不影响判定;Lead 补 5 处 non-blocking reader-comp。glossary+19/concepts+16/figures+6。
