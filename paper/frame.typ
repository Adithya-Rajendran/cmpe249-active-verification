#let ink = rgb("#172b35")
#let accent = rgb("#185a59")
#let muted = rgb("#455965")
#let hair = rgb("#b8c5c5")
#let sans = "Liberation Sans"
#let title-text = [When Can a Robot Record That an Object Is Gone?]
#let subtitle = [A Counterfactual Audit of Two-Image Scene Memory]
#let banner = [
  #text(font: sans, size: 24pt, weight: "bold", fill: ink)[When Can a Robot Record #linebreak() That an Object Is Gone?]
  #v(0.13em)
  #text(font: sans, size: 16pt, fill: accent, subtitle)
  #v(0.58em)
  #text(font: sans, size: 11pt, weight: "bold")[Adithya Rajendran]
  #h(0.8em) #text(size: 10.5pt)[San José State University]
  #h(1fr) #text(size: 8.5pt, fill: muted)[Revised 19 November 2026]
  #v(0.25em)
]
#let style(body) = {
  set document(title: "When Can a Robot Record That an Object Is Gone? A Counterfactual Audit of Two-Image Scene Memory", author: "Adithya Rajendran", keywords: ("robot scene memory", "object removal", "occlusion", "selective verification", "simulation"))
  set page(paper:"us-letter", margin:(left:0.9in, right:0.9in, top:0.78in, bottom:0.76in), header-ascent: 0.2in, footer-descent:0.22in,
    header: context if counter(page).get().first()>1 {set text(font:sans, size:8.2pt, fill:muted); [Rajendran #h(1fr) Counterfactual audit of scene memory #v(-0.3em) #line(length:100%, stroke:0.35pt+hair)]},
    footer: context { set text(font:sans, size:8.2pt, fill:muted); [#line(length:100%,stroke:0.35pt+hair) #v(-0.35em) November 2026 #h(1fr) #counter(page).display()] })
  set text(font: "Bitstream Charter", size: 10.6pt, fill: ink, lang:"en")
  set par(justify: true, leading:0.59em, spacing:0.63em)
  set heading(numbering:"1.1")
  show heading.where(level:1): set text(font:sans, size:14pt, weight:"bold", fill:ink)
  show heading.where(level:2): set text(font:sans, size:11pt, weight:"bold", fill:accent)
  show heading: set block(above:0.9em,below:0.37em)
  set figure(gap:0.4em)
  show figure.caption: set text(size:9.1pt)
  show figure.caption: set par(justify:false,leading:0.47em)
  show table.cell: set text(size:9.4pt)
  set link()
  body
}
#let aside(name, content) = block(width:100%, fill:rgb("#f1f5f3"),stroke:(left:1.7pt+accent),inset:(x:9pt,y:6pt))[
  #text(font:sans, size:9.35pt, weight:"bold",fill:accent)[#name] #h(0.22em) #content
]
