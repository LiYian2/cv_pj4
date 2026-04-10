你看到的“几何视角更正确”并不等于“相机或真实几何被显式监督修好了”；更可能是 pseudo RGB 监督把 3DGS 的可见性组织、外观分布和高斯生灭过程推向了一个更自洽的解，于是渲染出来的透视关系更顺了，但高频细节同时被平均化了。

先把最关键的一点说死：你当前 refine 里没有给 pseudo 分支做显式几何监督。 你的设计文档明确写了，Re10k-1 sparse 当前使用的是 target_depth = EDP fused depth、confidence_mask = normalized EDP confidence，但 target_depth 不直接作为 refine loss 的几何监督；pseudo loss 目前就是 confidence-weighted RGB L1：

𝐿
pseudo
=
∥
𝐶
⊙
(
𝐼
render
−
𝐼
target
)
∥
1
/
∥
𝐶
∥
1
L
pseudo
	​

=∥C⊙(I
render
	​

−I
target
	​

)∥
1
	​

/∥C∥
1
	​


也就是说，pseudo 分支在优化时“看的是颜色”，不是“看的是深度”。

但这不代表几何外观不会变。原因在于你的 refine 过程里，pseudo 不直接改几何，只是“不直接”而已，不是“几何完全不动”。 你现在的实现是 split backward：先对 pseudo branch 反传，清掉 pseudo 不允许更新的 param group 梯度，再对 real branch 反传，而且 real branch 还带 reg；文档里也明确写了这个效果是“pseudo 可仅改 appearance；real anchor 仍可更新几何”。换句话说，pseudo 虽然不直接推 xyz/scaling/rotation，但它先把 f_dc/f_rest/opacity 推到一个新状态，再由 real branch 在这个新状态上继续调整几何。

这就解释了为什么你会觉得“视角更对”。因为在 3DGS 里，人眼感受到的几何是否正确，不只取决于 xyz 坐标本身，还很大程度取决于：

哪些高斯还活着，哪些被 prune 了；
哪些高斯更透明、哪些更不透明；
边界和遮挡关系是不是更干净；
稀疏区域是不是被 densify 成了更连贯的结构。

而你的 refine 里这些东西都在动。当前主线配置 E 明确是 joint refine、freeze_geometry_for_pseudo=True、pseudo_trainable_params=appearance、densify_stats_source=real、lambda_pseudo=0.5。所以 pseudo 分支虽然只改 appearance，但 appearance 里是包含 opacity 的；同时 real 分支和 densify/prune 还在继续改 scene。 这意味着：pseudo RGB 先把“哪里该亮、哪里该暗、哪里该更透明/不透明”改了，随后 real branch 和 densify 又在这个新渲染分布上继续塑形。 最后你肉眼看到的“视角更正确”，其实常常是遮挡边界、轮廓位置、表面可见性组织更像 GT 了，而不一定是严格意义上的 3D geometry 被 pseudo supervision 直接学到了。

还有一个容易被忽略的点：你当前 pseudo target 是单边 left/right 二选一，不是 fused。run_pseudo_refinement.py 里加载 pseudo view 时，就是从 target_rgb_left.png 或 target_rgb_right.png 选一边来当 target。 单边 pseudo 往往会带着参考侧特有的修复偏差，但它仍然可能给出一个“全局结构上更顺”的颜色/可见性信号，于是 coarse layout 改善了；只是它的局部纹理不一定可靠。

然后说为什么图像质量反而更差、为什么会模糊。这里我觉得有四个直接原因。

第一，当前 pseudo supervision 本质上是“带 mask 的 RGB 拟合”，而不是“几何一致性约束下的 RGB 拟合”。 你的文档已经写了：depth 现在只表示可靠性，不做显式几何监督。 这就意味着，只要某块 pseudo RGB 在 mask 下被认为“值得信”，优化器就会努力把当前 render 拉向它；至于这块内容在 3D 上是否真的合理，当前 loss 没有直接约束。BRPO/PseudoView 论文恰恰提醒了这一点：diffusion-based pseudo view 往往“看起来清晰，但其实几何上不合理”，如果直接拿来优化，会把冲突信息写进重建，导致 artifacts 和 geometry degradation。

第二，L1 型 RGB loss 对不一致 target 的典型反应就是“取平均”，而平均的视觉结果就是糊。 你当前 L_pseudo 是 masked L1，不是 perceptual loss，也不是带结构项的损失。 当 pseudo target 本身含有修复偏差、参考侧偏差、或者单边 hallucination 时，最容易发生的事就是：模型为了同时兼顾 sparse real view 和 pseudo view，会把高频细节抹平，换取更低的逐像素误差。这样做经常会让 PSNR 变好，因为 misalignment 和局部噪声下降了；但视觉上你会觉得更“糊”、更“脏”。

第三，opacity 在你当前的 appearance preset 里是可训练的，这很可能是“几何看着更对、细节却更糊”的关键通道之一。 当前推荐配置把 pseudo_trainable_params 设成了 appearance，等价于 f_dc,f_rest,opacity。 f_dc/f_rest 改的是颜色和高频外观，opacity 改的是每个高斯在成像里“占多大权重”。当 pseudo target 不够可靠时，更新 opacity 很容易把局部边缘做厚、把细纹理做软、把遮挡轮廓做平滑。这样一来，轮廓可能更稳定、更接近 GT 的整体视角关系，但图像细节会变钝。

第四，densify/prune 虽然是 real-driven，但它仍然在放大当前 scene 状态的后果。 你的 refine 脚本会持续累计 densification stats，然后在指定迭代做 densify_and_prune()。 当前最优配置 E 里 final gaussians 也从几万长到了 42454。 即便 densify 统计来源设成 real，当前高斯场已经先被 pseudo 分支改过 appearance/opacity 了，因此 real branch 看到的 render 与梯度分布也已经不是初始那套。结果就是：densify/prune 往往会进一步强化当前这个“更平滑、更自洽”的解。它可能让结构更完整，但也可能把 blur 固化下来。

你文档里的指标其实已经在支持这个解释。当前最优 E 相比 baseline sparse GT，PSNR 和 SSIM 都明显更高，但 LPIPS 仍然更差：E 是 21.7658 / 0.7879 / 0.2110，baseline sparse GT 是 16.9367 / 0.6526 / 0.1722。 这非常像“对齐和整体结构改善了，但感知纹理质量没有同步改善”的典型模式。也就是说，你现在并不是“什么都变差了”，而是变成了一个更容易对上 GT 位置、但外观更像平均解的结果。

所以，把这个现象总结成一句话就是：

当前 refine 真正在做的，是用 pseudo RGB 去改 scene 的 appearance/visibility organization，再由 real anchor 和 densify 把这个组织固化；这会提升整体透视自洽和轮廓位置，但由于 pseudo target 还不够干净、loss 又是像素级 L1、且没有几何一致性强约束，高频纹理就会被平均掉，最后呈现为“几何视角更正确，但图像更糊”。

如果你接下来想验证这个判断，最能说明问题的不是再看一张图，而是做两个对照：
一组把 pseudo 可训练参数从 f_dc,f_rest,opacity 改成 f_dc,f_rest only；另一组保持不变。再比较边缘厚度、纹理锐度和 LPIPS 变化。这样能最快验证“opacity 是否在把 pseudo 偏差实体化”。

到底是谁把 pseudo 伪影写进了最终画面？是颜色特征，还是 opacity，还是两者耦合？

现有 run_pseudo_refinement.py 已经支持 csv-list，所以工程上其实很轻，只要加几个 preset 和更好的日志就行。

我建议改两处：

修改
run_pseudo_refinement.py

在 PRESET_PARAM_GROUPS 里新增：

appearance_no_opacity = {f_dc, f_rest}
appearance_dc_only = {f_dc}
appearance_sh_only = {f_rest}
appearance_plus_opacity = {f_dc, f_rest, opacity}

增加 grad logging
现在 history 只存 total / real / pseudo / reg / gaussian_count。
你应该再加：

每个 param group 的 gradient norm
每个 param group 的 update norm

这样你就能看到 pseudo 分支主要在推哪里。

实验上，我建议先做两组就够了：

f_dc + f_rest
f_dc + f_rest + opacity（当前 E）

如果前者视觉明显更干净，说明 opacity 很可能是“把脏纹理实体化”的关键通道。之后不是永久禁掉 opacity，而是可以做 staged unfreeze：比如前 1200 iter 不让 pseudo 动 opacity，后 800 iter 再放开。这才是“朝更好的方向优化”，而不是“参数越来越少”。

opacity 在你当前的 appearance preset 里是可训练的，这很可能是“几何看着更对、细节却更糊”的关键通道之一。 当前推荐配置把 pseudo_trainable_params 设成了 appearance，等价于 f_dc,f_rest,opacity。 f_dc/f_rest 改的是颜色和高频外观，opacity 改的是每个高斯在成像里“占多大权重”。当 pseudo target 不够可靠时，更新 opacity 很容易把局部边缘做厚、把细纹理做软、把遮挡轮廓做平滑。这样一来，轮廓可能更稳定、更接近 GT 的整体视角关系，但图像细节会变钝。