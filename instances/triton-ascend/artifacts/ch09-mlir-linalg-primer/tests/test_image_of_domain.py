"""t3 worked example：求像即子集，且滑窗算子的像比 tile 宽（"halo"）。

# PAPER: [Linalg §3.1] "The derivation of dense subsets is obtained by computing
the image of the iteration domain by the indexing function for each tensor."
(paper.md:L280-L284)
"""
from named_ops import make_conv_1d_nwc_wcf
from tiling import image_of_domain


def test_image_on_output_equals_the_tile_itself():
    """O 的索引映射对 (n,w,f) 全是纯恒等,像就是 tile 本身,不多不少。"""
    op = make_conv_1d_nwc_wcf()
    n, w, f, kw, c = (op.dim_names.index(x) for x in ("n", "w", "f", "kw", "c"))
    local_domain = {n: (0, 1), w: (8, 16), f: (0, 64), kw: (0, 3), c: (0, 32)}
    o_image = image_of_domain(local_domain, op.result_map)
    assert o_image == ((0, 1), (8, 16), (0, 64))


def test_image_on_input_is_wider_than_the_tile_conv_halo():
    """I 的空间轴索引是 w+kw(paper.md:L264 的耦合),tile 宽度为 8、核宽 kw 取
    满量程 (0,3) 时,I 需要读取的那一片宽度应为 8 + 3 - 1 = 10,而不是 8——
    这正是滑窗算子的"halo"([Linalg §3.1] paper.md:L280-L284 的直接推论)。
    """
    op = make_conv_1d_nwc_wcf()
    n, w, f, kw, c = (op.dim_names.index(x) for x in ("n", "w", "f", "kw", "c"))
    tile_w_lo, tile_w_hi = 8, 16
    local_domain = {n: (0, 1), w: (tile_w_lo, tile_w_hi), f: (0, 64), kw: (0, 3), c: (0, 32)}
    i_image = image_of_domain(local_domain, op.operand_maps["I"])
    n_range, w_range, c_range = i_image
    assert n_range == (0, 1)
    assert c_range == (0, 32)
    # w+kw 的像: 下界 = tile_w_lo + kw_lo, 上界 = (tile_w_hi-1) + (kw_hi-1) + 1
    assert w_range == (8, 18)
    tile_width = tile_w_hi - tile_w_lo
    image_width = w_range[1] - w_range[0]
    assert image_width == tile_width + 3 - 1  # 核宽 3 带来的 halo
    assert image_width > tile_width


def test_image_on_kernel_is_full_extent_when_kw_and_c_not_tiled():
    """K 只依赖 (kw,c,f);当这三维在 local_domain 里保持满量程时,K 的像就是
    K 的完整形状——kernel 不随空间 tile 变化,这也是"归约维本参考实现不切"
    这条范围收窄（见 tiling.py 文档）在具体数字上的体现。"""
    op = make_conv_1d_nwc_wcf()
    n, w, f, kw, c = (op.dim_names.index(x) for x in ("n", "w", "f", "kw", "c"))
    local_domain = {n: (0, 1), w: (8, 16), f: (0, 64), kw: (0, 3), c: (0, 32)}
    k_image = image_of_domain(local_domain, op.operand_maps["K"])
    assert k_image == ((0, 3), (0, 32), (0, 64))
