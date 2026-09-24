"""What each case tests. Kept away from `cases.py` so a driver never sees the answer.

  plain      a correct result exists and the direct approach reaches it
  trap       the obvious first move is wrong (bad box, shared hue, out-of-sync colour key,
             no hue to select by)
  ambiguous  the request does not determine which object is meant
  refusal    the operation will damage the picture and the tool should say so
"""
KIND = {
    'door': ('trap', 'the postbox beside the door is the same red, so a hue-based recolour '
                     'takes both and the request forbids exactly that'),
    'cat_cutout': ('plain', 'one clear subject against a flat backdrop'),
    'taxi': ('trap', 'the taxis are small and far, and the street carries other yellows '
                     '(signs, lights, road markings)'),
    'cones': ('trap', 'a row of thin cones: any box around one is mostly road, and there '
                      'are a dozen of them'),
    'two_cars': ('ambiguous', 'two red cars overlap in the frame; "the red car" does not '
                              'say which'),
    'heatmap': ('trap', 'the colorbar is the key to the cells -- re-theme the cells alone '
                        'and the figure carries two colour scales'),
    'zipf': ('plain', 'a published two-panel figure with a default palette'),
    'cat_erase': ('refusal', 'the cat is most of the frame; inpainting has nothing to '
                             'reconstruct from and the tool warns about exactly this'),

    'bird_wire': ('plain', 'a small, sharply separate subject on a flat sky -- inpainting '
                          'has real texture around it to work from, unlike cat_erase'),
    'powerlines': ('refusal', 'the wires cross the whole frame and split into a lattice of '
                              'thin lines; there is no "behind them" to reconstruct and no '
                              'box captures only wire'),
    'lamppost': ('trap', 'a thin vertical object against open sky -- a box around it is '
                        'mostly sky, the case tools.py itself warns about ("a lamp post"); '
                        'the right move is a point prompt, not a box'),
    'lighthouse': ('plain', 'a whole-image crop with no segmentation involved -- a cheap '
                            'sanity check that the loop does not over-reach for a plain '
                            'geometric request'),
    'white_car': ('trap', 'a white subject has no meaningful hue to select by; the default '
                          'hue-window isolate cannot touch it at all, and the correct move '
                          'is oklab with a tight tolerance or an object-based isolate'),
    'two_dogs': ('plain', 'two distinct, unambiguous dogs the request names together -- '
                         'the correct call is recolour_object with boxes=[...], not two '
                         'separate calls or a guess at which one is meant'),
    'two_bicycles': ('plain', 'one bicycle against a leafy, textured background -- tests '
                             'isolate_object (not isolate_colour) as a positive control, '
                             'since the shared green/brown hues would defeat a hue isolate'),
    'two_cars_multi': ('plain', 'the SAME frame as two_cars, but "both red cars" resolves '
                               'the ambiguity that made the singular request an honest '
                               'stop -- the pair only means something read together'),
    'chart_bar': ('plain', 'a second, independent test of "default matplotlib colours -> '
                          'brand palette" on a bar chart rather than zipf\'s line plot'),
    'chart_scatter': ('trap', 'a colorbar keyed to the point colours, structurally like '
                             'heatmap but on a scatter figure -- re-theme the points alone '
                             'and the key goes out of sync'),
    'legend_bleed': ('trap', 'the legend box sits ON TOP of two of the four lines with '
                            'framealpha=0.75; cleaning it can only keep the swatches and '
                            'text, and an over-large box would eat real curve data'),
    'annotation_box': ('trap', 'a semi-transparent annotation callout sits directly over '
                              'the curve and its fill, at the peak the label refers to -- '
                              'the box must go, the curve underneath and the label/arrow '
                              'must not'),
}
