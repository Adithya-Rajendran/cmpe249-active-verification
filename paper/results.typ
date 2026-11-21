#let summary = json("evidence/summary.json")
#let eng(k) = summary.engines.at(k)
#let env(k, v) = eng(k).environments.at(v)
#let methods = ("homography", "dense", "registered")
#let label = ("Homography", "Dense geometry", "Qwen (diagnostic)")
#let names = ("Isaac Sim", "MuJoCo")
#let engines = ("isaac", "mujoco")
#let ratio(value,total) = [#value/#total]
#let point(v) = {
  let n=calc.round(v,digits:1)
  if n < 0 { [−#str(calc.abs(n))] } else if n > 0 { [+#str(n)] } else { [0] }
}
#let interval(values) = [#point(values.at(0)) to #point(values.at(1))]
#let empty_or_count(n,total) = if total==0 { [—] } else { ratio(n,total) }
#let head(value) = table.cell(fill:rgb("#e9f0ee"), text(weight:"bold",value))
#let rowline = table.hline(stroke:0.45pt+rgb("#b8c5c5"))
#let cstroke = (x: none, y: none)

= Results

== What carried over from the exploratory cohort?

On the ten layouts used to select the comparison, dense geometry recognized all 25 clear tabletop removals at the largest motion in each engine. The homography recognized 17 in Isaac Sim and 16 in MuJoCo. Across all 1,200 decisions, geometry gained 113 correct answers and lost 70. Two newly wrong decisions declared a retained foam brick removed at the same shelf stop with the blocker parked and placed. On one fully blocked shelf pair, the homography gave the same removal answer twice (@example). The retained and removed scenes produced identical inputs. These observations motivated the comparison. The main cohort tests the same operating points on different target objects and layouts.

Registered Qwen passed the archived clear-view screen. It answered PRESENT on 36/36 unchanged references, PRESENT on 40/40 clear retained queries and REMOVED on 40/40 clear removals. Each engine contributed 20 of each clear class across the nominal and moved cameras. All answers passed the parser. The median reference box covered 9,496.5 pixels in the screen and 6,666 among the main inputs that reached Qwen, measured on the 1200 × 675 inputs. This screen established performance on its clear controls. It did not qualify this reader for abstention under occlusion.

== Clear removals after the camera moved

#figure(table(columns:(1.05fr,1fr,1fr,1.85fr),align:(left,center,center,center),stroke:cstroke,inset:(x:4pt,y:4pt),
  head([Simulator]),head([Homography]),head([Dense geometry]),head([Dense − homography #linebreak() pp (95% interval)]),
  ..engines.enumerate().map(((i,k))=>{
    let p=env(k,"all").primary
    (names.at(i),ratio(p.homography_correct,p.n),ratio(p.dense_correct,p.n),
      [#point(p.delta_pp) (#interval(p.delta_interval_pp))])
  }).flatten(),rowline),
  kind:table,caption:[Correct REMOVED decisions with the blocker parked after a 6 cm translation and 3° yaw. Each engine contributes the same 16 arrangements and 80 queries. Abstentions count in the denominator. Percentage-point intervals resample complete arrangements within environment.]) <primary-table>

#figure(image("figures/primary-by-target.png",width:97%),caption:[Correct removal decisions out of twenty after camera motion, split by environment and target. Each group contains four layouts and five camera stops per layout. Offset markers keep equal counts visible.]) <target-figure>

#include "primary-finding.typ"

== Definite decisions and physical state

#figure(table(columns:(1fr,1.3fr,0.95fr,1.05fr,1fr,1fr),align:(left,left,center,center,center,center),stroke:cstroke,inset:(x:4pt,y:3.5pt),
  head([Simulator]),head([Verifier]),head([Correct / 640]),head([Definite / 640]),head([False removal / 320]),head([False presence / 320]),
  ..engines.enumerate().map(((i,k))=>{
    let a=env(k,"all").overall
    methods.enumerate().map(((j,m))=>(if j==0 {names.at(i)} else {[]},label.at(j),str(a.at(m+"_correct")),str(a.at(m+"_definite")),str(a.at(m+"_false_removal")),str(a.at(m+"_false_presence")))).flatten()
  }).flatten(),rowline),
  kind:table,caption:[All interventions, both motion levels. Each engine has 320 retained and 320 removed cases. A false removal would delete a retained target's memory record. A false presence would keep a stale record. Correct and definite counts use all queries.]) <counts-table>

The registered language model recognized 80/80 clear moved-camera removals in Isaac and 79/80 in MuJoCo. It also reported removal on 24/320 retained Isaac queries and 27/320 retained MuJoCo queries (@counts-table). Eleven of these false removals per engine occurred with the blocker parked. In every parked-blocker retained view, the native mask covered at least 99.96% of the isolated target silhouette. All the model's wrong definite decisions were removals. Across both engines it gave 781 correct answers, compared with 776 under homography and 797 under dense geometry. All 3,840 query evaluations completed without a technical execution failure.

Across the 640 actual removals, the dense rule missed 241 and registered Qwen missed 221. Both methods abstained on every missed removal. Across the 640 retained queries, the same methods falsely deleted two and 51 targets, respectively.

A rule that always abstains makes zero errors and zero correct decisions on this panel. A rule that always reports removal achieves 80/80 on the primary endpoint and 320/640 overall, with 320 false removals per engine. The two controls explain why resolution, coverage and wrong commitments must be reported together.

== Answers by target visibility

For this post hoc analysis, we assigned both members of each pair the fraction of the isolated target silhouette exposed in the retained revisit. The removed member has no target pixels of its own. Among 640 pairs, 176 had zero exposure, 402 had complete exposure, and four more had at least 99.95% exposure. The other 58 pairs covered the intervening range. Each simulator rendered the same 16 layouts.

#figure(image("figures/answers-by-visibility.png",width:100%),caption:[Physical-state decisions by exposure of the retained twin. Each row gives the exposure band and query count per state, pooled across engines. Green denotes correct decisions, rust wrong decisions, and gray UNKNOWN. Counts include every query in the band. At 0% exposure every definite answer came from a MuJoCo pair with identical inputs, so each green segment in that row is a guess that matched the physical state. The post hoc exposure bins describe this panel.]) <visibility-figure>

At 50% to below 90% exposure, 28 pairs came from five layouts (@visibility-figure). Homography identified 19/28 retained targets and 12/28 removals, with one false deletion. Dense geometry identified 1/28 retained targets and 3/28 removals. It abstained on the other 52 queries. The clear-removal gain therefore did not carry through to this partial-exposure band.

== When opposite states gave identical inputs

#figure(table(columns:(1.1fr,1fr,0.8fr,1fr,1fr,1fr),align:(left,center,center,center,center,center),stroke:cstroke,inset:(x:4pt,y:3.5pt),
  head([Simulator]),head([Exact / all pairs]),head([Layouts with exact pairs]),head([Homography commits]),head([Dense geometry commits]),head([Qwen system commits]),
  ..engines.enumerate().map(((i,k))=>{
    let c=eng(k).certificates
    (names.at(i),ratio(c.exact,c.pairs),str(c.layouts),..methods.map(m=>empty_or_count(c.at(m).commit,c.exact)))
  }).flatten(),rowline),kind:table,
  caption:[Exact equality covers both decoded images, the reference box, and the fixed task within each method. The last three columns count pairs receiving a definite decision from the complete system. The Qwen registration gate passed only four exact pairs. A dash denotes an empty comparison set.]) <ambiguity-table>

#include "certificate-finding.typ"

== Trained STILL versus zero-shot Qwen (retrospective)

On the archived panel, the trained crop verifier detected 462 of 640 removals and zero-shot Qwen detected 419 (@still-results). The difference in missed-removal rates, STILL minus Qwen, was −6.7 percentage points (paired 95% layout-bootstrap interval, −11.6 to −2.2). The corresponding false-deletion difference was −2.3 points (−8.6 to +4.2). These descriptive intervals hold the development threshold fixed. The 16 layouts support a clearer difference in missed removals than in false deletions.

#let counts = csv("evidence/still-comparison-counts.csv", row-type: dictionary).filter(r => r.subset == "full" and r.engine == "both")
#let exact = csv("evidence/still-exact-pairs.csv", row-type: dictionary)
#let variants = (
  ("homography", "Homography"),
  ("dense", "Dense geometry"),
  ("registered_qwen", "Zero-shot Qwen"),
  ("still_place", "STILL, registered crop"),
  ("still_place_conservative", "STILL, conservative crop"),
  ("still_fused", "STILL, scene plus crop"),
)
#figure(table(columns:(1.65fr,1fr,1fr,1fr),align:(left,center,center,center),stroke:cstroke,inset:(x:4pt,y:3pt),
  head([System]),head([Missed removals #linebreak() / 640 removed]),head([False deletions #linebreak() / 640 retained]),head([Deleted both states #linebreak() / 88 exact pairs]),
  ..variants.map(((key, name)) => {
    let r = counts.find(r => r.method == key)
    let p = exact.find(r => r.method == key)
    (name, r.missed_removals, r.false_deletions, p.delete_both)
  }).flatten(),rowline),kind:table,
  caption:[Deletion decisions on the same 1,280 queries. Each simulator contributed 320 retained and 320 removed queries. Every missed or withheld deletion on a removed target counts as a miss. Only MuJoCo produced exact opposite-state pairs.]) <still-results>

The original Qwen alignment check admitted 419 removed and 415 retained queries. Within those 834 queries Qwen found all 419 removals but erased 51 retained targets. STILL found 405 and erased one. Among the 446 queries the original gate rejected, STILL found another 57 removals and erased 35 retained targets. Overall, missing crops accounted for 153 of STILL's 178 missed removals. The comparison includes differences in registration, prompts and thresholds as well as weights.

The error patterns also differed. Qwen's 51 false deletions comprised 47 fully exposed targets and four fully hidden targets. STILL's 36 comprised 28 fully hidden targets and eight partly visible targets. Of the 88 identical pairs, four reached the original Qwen reader and it deleted both states on all four. Pi3 supplied usable crops on 16 pairs. STILL deleted both states on 11, withheld deletion below threshold on five and lacked a usable crop on the remaining 72. The scene-plus-crop rule deleted both states on 79 pairs (@still-results).

== What did the fixed-camera calibration measure?

#include "calibration-finding.typ"

== How sensitive were the failures to the camera and room?

Three follow-up panels used the frozen verifiers. On the shelf, four previously identified MuJoCo failures recurred at the original camera angles. When the shelf edge was leveled in the image by rotating each camera at the same position, registration stopped all four pairs before the language model ran. Eight fresh service-corridor layouts then tested a bottle or a drill behind a sliding door. The declared oblique test found zero double-REMOVED calls on eight exact MuJoCo pairs. A secondary frontal view gave seven on eight exact pairs. That view also changed camera position and heading.

The furnished workshop repeated those same eight corridor layouts with a corrected sliding-door bay. Its declared frontal test gave zero double-REMOVED calls on eight exact MuJoCo pairs. Registration stopped all eight. Changing the room altered several visual cues together. On open-door controls the simple homography classified all 80 queries correctly in each engine and abstained on every closed-door query. The registered language model detected all 40 actual open-door removals in each engine, yet falsely removed an exposed drill on four of 40 retained queries in Isaac and two of 40 in MuJoCo. All six errors occurred on the two added robot camera mounts. The supplement provides the pair tables for each panel. The corridor and workshop share layouts, so their counts describe a matched sensitivity test.
