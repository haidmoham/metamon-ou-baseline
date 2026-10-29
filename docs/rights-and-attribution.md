# Rights, provenance and scope audit

Checked 2026-10-29 against immutable revisions in `pins.json`. This is a technical provenance record, not a guarantee about every underlying training-data right.

| Component | Evidence / stated license | Treatment here |
| --- | --- | --- |
| Metamon source | [MIT LICENSE at pinned source](https://github.com/UT-Austin-RPL/metamon/blob/0a00a759c9a4382a2877088d828302ec294a05a5/LICENSE); copyright 2025 UT Robot Perception and Learning Lab | Download original source; retain its license; no simulator/framework rewrite |
| AMAGO 3.4.0 | [MIT source](https://github.com/UT-Austin-RPL/amago/tree/0974781a9096ff43df1b708312256f96fc2ab127) | Dependency, source attribution retained |
| UT-Austin poke-env fork | [MIT source](https://github.com/UT-Austin-RPL/poke-env/tree/e1268d270c3f2bd32c7ff5713e01062302020579); original copyright hsahovic | Dependency; replay template loaded from installed package, not rebranded |
| Pokémon Showdown | [MIT source](https://github.com/smogon/pokemon-showdown/blob/d62d3a39878f7ef1c6e1cfbd84e8feb2ce7a9fdd/LICENSE) | Pinned upstream simulator and its npm lock; original notices retained |
| Kakuna checkpoint | [Hub model card at pinned revision](https://huggingface.co/jakegrigsby/metamon/blob/ac8e88e7da8dc2a2b85176d88f81fd8854b1445b/README.md) declares Apache-2.0 | Download directly from original publisher, verify hash; **no redistribution here** |
| Competitive team archive | [Pinned dataset repository](https://huggingface.co/datasets/jakegrigsby/metamon-teams/tree/bff785c80a06273faee1c675d368948f5dd65d44) | Dataset metadata checked had no license declaration; upstream says competitive teams are often human-made/Smogon-derived. Local download for upstream inference only; **do not publish copies or assume blanket sublicensing** |
| FlashAttention | [BSD-3-Clause source, version 2.7.4.post1](https://github.com/Dao-AILab/flash-attention/tree/v2.7.4.post1) | Optional GPU runtime dependency; installation untested in CPU cloud environment |
| Pokémon names, characters, artwork and branding | Not licensed by this repository's MIT license | No artwork or branding assets bundled; no endorsement claimed |
| Training demonstrations and self-play datasets | Separate upstream datasets and provenance | Not downloaded or redistributed; no new training/fine-tuning performed |

The model-card license is evidence of the publisher's license statement, not proof of rights in every underlying human replay, third-party game element or dataset. Even where redistribution may be possible under license conditions, this baseline deliberately avoids redistributing checkpoints. A future request to redistribute weights or datasets requires a separate scoped review of notices, license text and provenance.

## Credit

Metamon was developed by Jake Grigsby, Yuqi Xie, Justin Sasek, Steven Zheng and Yuke Zhu at UT Austin RPL. Its pretrained models and battle interfaces do the substantive learning and Pokémon representation work. AMAGO provides the policy implementation. Pokémon Showdown supplies the simulator; poke-env supplies transport/environment integration. This repository adds setup pins, a thin sequential evaluation adapter and evidence handling.

```bibtex
@misc{grigsby2025metamon,
  title={Human-Level Competitive Pokemon via Scalable Offline Reinforcement Learning with Transformers},
  author={Jake Grigsby and Yuqi Xie and Justin Sasek and Steven Zheng and Yuke Zhu},
  year={2025},
  eprint={2504.04395},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2504.04395}
}
```

Any ratings quoted by upstream belong to their experiments. This project has not reproduced them and does not claim them as its own performance.
