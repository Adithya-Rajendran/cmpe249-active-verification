= Discussion

#include "interpretation.typ"

This audit addresses the decision to update a specified remembered location from two RGB images. A robot could instead maintain a posterior over states, use an inventory prior, or request another view. Those options use additional information or a different decision objective. For the interface tested here, a physically correct guess on an exact pair still gives no evidence that the remembered location is empty. Reporting how often the system abstains on those pairs together with its clear-removal resolution exposes that distinction.

The evaluation uses two chosen target meshes, two blocker families and one lighting condition per engine. Its static removal intervention suppresses the target in the render. Changes that would follow from touching or displacing other objects require a dynamics experiment. The engines share the scene geometry and camera trajectories. The two renders therefore reveal sensitivity to rendering within the same design. Object additions, relocation, deformable occluders, physical cameras, calibrated real-world risk and end-to-end planning remain outside the observed evidence. The sample can identify individual failure mechanisms and quantify their frequency on this panel. Sixteen layouts give limited precision for differences between environments or object types.

The exact certificate is deliberately narrow. Different pixels can result from sampling, shadows, other optical effects, or informative target evidence. Distinguishing these effects requires evidence beyond a target silhouette. The calibration tests a local photometric cue at a fixed pose. Semantic support on partially occluded or moved-camera judgments remains unresolved. A subsequent active test could start from ambiguous pairs on fresh layouts and allow one additional view. Compare a selected viewpoint with fixed and random choices at equal view and movement budgets. Count correctly resolved states and new false removals, keeping every unresolved pair in the denominator.

#heading(level:2,numbering:none)[Data and reproducibility]
The accompanying supplement contains the local prospective protocol, target assignments, case and pair tables, qualification and calibration measurements, source, and an independent recount. The image archive contains every delivered RGB input with a content hash. Native outputs preserve masks and the recorded scene state for evaluator-side inspection. The study runs entirely in simulation. The protocol carries a local timestamp recorded before generation of the new layouts. Scoring rules and the comparison were selected during earlier development.

#heading(level:2,numbering:none)[AI assistance]
Codex and Terra assisted with experiment implementation, simulator operation, computational review and manuscript preparation. Simulator interventions supplied the physical-state labels. Exact-input checks and the calibration used programmatic measurements.
