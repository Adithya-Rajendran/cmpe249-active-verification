# Locate the remembered object

Use only your assigned packet. Each case contains a boxed earlier reference and an unmarked revisit.
The object can remain in its original location or leave the scene. The camera can move.
The upper panels show both complete images. The detail panels use the same crop and full-image pixel coordinates.

Inspect each pair independently. Identify the boxed object in the reference.
Find stable scene landmarks that locate the same place in the revisit.
Use two landmarks when available. Describe their relationship to the remembered object and any camera shift.

Estimate the rectangle that the entire object would occupy in the revisit if it remained at its remembered position.
Include the parts a foreground surface could cover. Use null if the landmarks cannot give you a defensible rectangle.
Describe how uncertain the location is.

Write a JSON array to `localization.json`. Give one entry per case with `id`, `target`, `landmarks`, `box` and `uncertainty`.
Use nonempty text for each description. Write the box as four integer coordinates in the full 1200 by 675 revisit.
Use left, top, right and bottom, with exclusive right and bottom edges. Submit only the localization at this stage.
