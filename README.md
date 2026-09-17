# Revisiting Jigsaw Objectives for Fine-Grained Visual Representations

Self-supervised jigsaw objectives, implemented as drop-in losses for
[DINOv3](https://github.com/facebookresearch/dinov3). A crop is cut into tiles, the tiles are shuffled into a mosaic, and the
student is asked something about the arrangement: the direction between two tiles, the cell a
tile came from, the region a small crop was taken from, or which of 1000 fixed permutations
produced the mosaic. A gap is dropped between tiles so no two of them continue each other and
the puzzle cannot be solved by matching edges.

The same folder also contains two objectives we compare against: NeCo, which matches the
teacher's patch nearest-neighbour ordering, and boundary-centric masked modelling, which makes
the teacher predict a line-segment field, validates it, and forces the tokens it crosses into
the mask. Every objective is a loss plus a head, added to the usual DINO and iBOT terms; the
backbone, the augmentations and the EMA teacher are unchanged.

```
views.py         tile cut, shuffle, mosaic, block pooling
relpos.py        8-way relative position; block_size > 1 gives the global-puzzle variant
abspos.py        absolute position with annealed target smoothing
region.py        region prediction from an unpooled crop
permutation.py   1000 fixed permutations as classes, and the generator for the sets
neco.py          patch nearest-neighbour ordering over the crop overlap
boundary.py      boundary field, segment decoding, a-contrario validation, categorical loss
selftest.py      runs every objective on random tensors
configs/         one file per objective, matching the runs in the paper
permutations/    the stored permutation sets for the 3x3, 4x4 and 12x12 grids
```

DINOv3 itself is not included here. Clone it separately; these files add the objectives to it
and change nothing else in the framework.

```bash
git clone https://github.com/facebookresearch/dinov3
cp -r jigsaw dinov3/            # the objectives import only torch, torchvision and numpy
```

## Setup

```bash
conda create -n jigsaw python=3.10 -y && conda activate jigsaw
pip install -r requirements.txt
python selftest.py
```

`selftest.py` needs no data and no GPU. It builds each head, runs a forward and a backward
pass, and checks the losses start at chance: `ln(8)` for the direction task, `ln(1000)` for the
permutation task, `1/144` accuracy for absolute position.

## Run

The objectives attach to a DINOv3 training loop. Each iteration, build the view from the
jigsaw crop, pool the mosaic tokens into tiles, and add the loss to the total:

```python
from views import build_view
from relpos import RelPosHead, RelPosLoss, relpos_loss

view = build_view(crops, grid=12, tile_patches=2, gap_patches=1, patch_size=16)
student_tokens = student.backbone(view["mosaic"], is_training=True)["x_norm_patchtokens"]
teacher_tokens = teacher.backbone(crops, is_training=True)["x_norm_patchtokens"]

logits, target, valid = relpos_loss(student_tokens, teacher_tokens, view, head, block_size=1)
total = total + 0.3 * RelPosLoss()(logits, target, valid)
```

The teacher tokens come from the intact crop, so the targets never depend on the shuffle. For
the refinement runs the teacher is frozen at iteration 100k and refreshed every 1250 steps
(`frozen_teacher` in the configs); the permutation and NeCo objectives need no teacher tiles.

Settings per experiment are in `configs/`. The tile-size study varies the cut while keeping the
576px view and the 1-patch gap fixed:

```
grid  tile  tiles  mosaic
18    1     324    288px
12    2     144    384px
 9    3      81    432px
 6    5      36    480px
 4    8      16    512px
 3   11       9    528px
```

## Acknowledgements

The training framework is [DINOv3](https://github.com/facebookresearch/dinov3) (Siméoni et al.,
[arXiv:2508.10104](https://arxiv.org/abs/2508.10104)); the backbone, augmentations, DINO and iBOT
losses, gram anchoring and the EMA teacher are theirs, and this code is subject to the DINOv3
License Agreement in `LICENSE.md`.

The objectives we compare against are reimplementations of published work:

- `permutation.py` follows Noroozi & Favaro, *Unsupervised Learning of Visual Representations by
  Solving Jigsaw Puzzles* ([arXiv:1603.09246](https://arxiv.org/abs/1603.09246)), including the
  greedy maximal-Hamming permutation set of their Algorithm 1.
- `neco.py` follows Pariza et al., *NeCo: Improving DINOv2's Spatial Representations with Patch
  Neighbor Consistency* ([arXiv:2408.11054](https://arxiv.org/abs/2408.11054)), using the
  differentiable sorting networks of Petersen et al. (`diffsort`).
- `boundary.py` follows LingBot-Vision, *Pushing the Boundaries in Vision Pretraining*
  ([arXiv:2607.05247](https://arxiv.org/abs/2607.05247)). The boundary field is the
  holistically-attracted field of Xue et al., and the validation is the a-contrario test of
  Desolneux et al. as used by LSD (Grompone von Gioi et al.).

Two deviations from LingBot are ours and are marked in the code: its corner detector is not
released, so we use Shi-Tomasi corners, and the level-line guidance that bootstraps the field is
annealed off over the first 20k iterations.
