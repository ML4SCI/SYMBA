from symbolic_jepa.tokenizer import PrefixTokenizer, sympy_to_prefix, prefix_to_sympy
from symbolic_jepa.expressions import Expression, VarMeta, load_feynman_csv, load_synthetic_pkl
from symbolic_jepa.encoder import TNet
from symbolic_jepa.decoder import SymbolicTransformer
from symbolic_jepa.jepa import (
    JEPAPredictor, IdentityPredictor, jepa_loss, subsample_consistency_loss,
)
from symbolic_jepa.dataset import (
    PointCloudDataset, MultiViewPointCloudDataset,
    sample_pool, subsample_and_normalize, sample_and_normalize,
    build_feynman_splits, build_synthetic_splits,
    build_multiview_synthetic_splits,
)
from symbolic_jepa.templates import (
    ConstantTemplate, ConstantSampler, DynamicConstantPointCloudDataset,
    DynamicConstantMultiViewDataset, InstantiatedExpression, templatize,
    AUGMENTATION_VERSION, augmentation_seed, templates_fingerprint,
    build_templates_from_strings, build_template_splits,
    build_multiview_template_splits,
    save_template_dataset, load_template_dataset,
    canonical_split_report, describe_realizations, audit_constant_sampling,
)
from symbolic_jepa.evaluation import (
    r2_score, teacher_forced_accuracy, teacher_forced_counts,
    build_prefix_tree, branching_mask, fit_constants,
    equations_equivalent, evaluate_predictions, cleanup_eval_pool,
)
from symbolic_jepa.diagnostics import (
    sym_spread, pred_spread, retrieval_top1, common_mode,
    view_consistency_diagnostics, generate_diagnostic_embeddings,
)
