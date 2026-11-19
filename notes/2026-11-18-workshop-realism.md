# Superseded furnished workshop

During scoring, visual review identified a real collision in the first furnished room. The 95 cm door slid 98 cm sideways into a solid corridor wall. The renderer placed the two endpoint poses directly. Image, mask and camera checks passed even though the room could not accommodate the motion. [Original rendering](figures/2026-11-18-workshop-realism/appearance-comparison.jpg) · [Swept-volume comparison](figures/2026-11-18-workshop-clearance/door-clearance.json)

The frozen verifiers completed 960 decisions in this room. The primary MuJoCo frontal error occurred on zero of eight exact pairs. These scores document the rendering fault investigation. The corrected study uses its own geometry, blinded packet, scores and robotics acceptance tests. [Archived diagnostic counts](figures/2026-11-18-workshop-realism/diagnostic-summary.json) · [Corrected study](2026-11-18-workshop-clearance.md)
