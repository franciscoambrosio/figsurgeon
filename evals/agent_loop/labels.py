"""What the final pictures actually show, judged by looking at every one of them by
someone who did not drive the run. The drivers' own closing accounts are recorded, not
believed: a third field on widened-run entries notes whether that account matches what
the picture shows, or how it oversold/undersold the result.

  correct       the picture is what the user asked for, small bleed aside
  wrong         it is not, whatever the verdict said
  honest stop   no edit, and not editing was the right call
  incomplete    no edit, and not a considered stop -- ran out of turn budget mid-attempt
                with no acknowledgement that nothing was delivered

Keys are the bare case id for the original eight (model unrecorded) and (case_id, model)
tuples for runs through `run_models.py`.
"""
LABEL = {
    'door': ('correct', 'the door is dark green, the postbox is still red; a small wall '
                        'bracket attached to the door frame went green too'),
    'cat_cutout': ('correct', 'clean RGBA cutout, ears and all four paws, no backdrop left'),
    'taxi': ('correct', 'the taxis are yellow and the street is grey; one orange billboard '
                        'high on the left keeps its colour, which is a real orange sign'),
    'cones': ('correct', 'every cone is blue, white bands intact; a little blue on two '
                         'fuel-pump stripes'),
    'two_cars': ('honest stop', 'two identical red estates; the driver said which one is '
                                'not determined and edited nothing'),
    'heatmap': ('correct', 'cells and colorbar both on the brand ramp, the diagonal '
                           'structure unchanged'),
    'zipf': ('correct', 'both panels remapped, the red fit line deliberately left as the '
                        'one thing that is not colormap content'),
    'cat_erase': ('honest stop', 'the erase left a cat-shaped ghost; the driver looked, '
                                 'undid it, and said inpainting cannot fill that region'),

    # ---- widened run, z-ai/glm-5.3-flash (primary model, all 20 cases) ----
    ('heatmap', 'z-ai/glm-5.3-flash'): (
        'correct', 'cells and colorbar both on the brand ramp in one call, diagonal '
                  'structure unchanged', 'matches -- said exactly this'),
    ('cat_cutout', 'z-ai/glm-5.3-flash'): (
        'correct', 'clean RGBA cutout, ears/whiskers/paws/tail intact; a faint blue-grey '
                  'fringe survives on the outline but does not change the answer',
        'matches -- said the mask covered only the cat and the edges were checked, which '
        'is true; did not mention the faint fringe, a minor oversell'),
    ('lamppost', 'z-ai/glm-5.3-flash'): (
        'correct', 'a tight box around just the lamp head avoided the trap (mask preview '
                  'confirms no pole/sky in the segmented region); diffuser keeps a faint '
                  'warm tone, pole and sky both read neutral grey',
        'matches -- said the lamp stayed in colour and the rest went grey, which is true'),
    ('taxi', 'z-ai/glm-5.3-flash'): (
        'correct', 'RE-RUN at max_turns=20 (the 10-turn attempt is kept as '
                  'logs/taxi.10turn__z-ai_glm-5.3-flash.json and out/taxi.10turn__...png, '
                  'and was: incomplete -- the picture completely UNEDITED after ten calls, '
                  'every one a sample_colour or zoom, not one landing a clean reading on a '
                  'taxi). The extra budget bought the edit: the same hunt for a sample runs '
                  'for six calls, then a zoom maps the taxi back to original coordinates, '
                  'one lands RGB(242,193,61), three preview_colour_mask calls walk the '
                  'tolerance (0.06 catches a sunlit facade, 0.04 is tight, 0.045 chosen), '
                  'and isolate_colour keeps 1.8 % of the frame. Every taxi is solid yellow '
                  'including roof signs and shaded flanks, the street, buses, the red '
                  'MUSEUM OF ART billboard and the white cars are all clean grey. Two warm '
                  'patches survive on an out-of-focus building top right, which the driver '
                  'saw and attacked with a tan replace_colour -- that left salmon halos on '
                  'the taxi roofs, and it UNDID it rather than ship them. Same class of '
                  'leftover as the original run\'s orange billboard',
        'UNDERSELL BY OMISSION, and the mirror image of the 10-turn run: the picture is '
        'right and the transcript ends mid-plan ("I\'ll undo it and fix the two escapee '
        'patches locally instead") with no statement that a good colour-pop is what got '
        'delivered. Out of turns, not out of things to say -- turns_used 20 of 20, '
        'stop_reason never max_tokens, which the log now RECORDS rather than leaving to '
        'arithmetic'),
    ('chart_bar', 'z-ai/glm-5.3-flash'): (
        'wrong', 'RE-RUN at max_turns=20 (the original 8-turn and 10-turn attempts were '
                'both incomplete -- unedited -- see the session notes for the full '
                'tool-routing story: replace_colour correctly diagnosed as unable to '
                'reach a same-hue different-lightness target, then recolour_object). At '
                '20 turns it got much further -- Sales bars are dark navy, Growth bars '
                'light blue, legend swatches match -- but the delivered picture still has '
                'two visible defects: the North Sales bar keeps a sliver of the original '
                'tab:blue peeking out at its base/edge, and a thin strip of the original '
                'orange survives along the bottom of all four Growth bars. Close, '
                'genuinely most of the way there, but not what was asked for cleanly, so '
                'wrong rather than correct',
        'transcript ends mid-sentence ("Setting preserve_shading=false because chart '
        'bars are flat fills:") with no closing summary and no acknowledgement of the '
        'leftover slivers -- MISLEADING BY OMISSION, same pattern as the rest of this run'),
    ('cat_erase', 'z-ai/glm-5.3-flash'): (
        'wrong', 'an edit WAS made and it is visibly bad: erase_object correctly came back '
                'verified=False (ghost), the driver undid it and used generative_fill as '
                'the documented fallback, which left a crisp, differently-coloured outline '
                'of the whole cat against the podium -- as recognisable as the original '
                'ghost. The driver noticed this itself ("a thin dark line traces the cat\'s '
                'old outline") and spent its remaining calls zooming to map the line for '
                'paint-out, but ran out of its 10-turn budget before making a single '
                'corrective call. Not requeued at 20 turns: per the coordinator\'s '
                'definition, incomplete means no bad edit was made, and a visible cat '
                'silhouette shipped to the user is a bad edit whatever the reason it '
                'was not fixed',
        'MISLEADING BY OMISSION -- driver_said ends "Let me zoom on the remaining '
        'sections to trace precise coordinates for paint-out," giving no hint that the '
        'delivered picture still shows an obvious cat outline'),
    ('zipf', 'z-ai/glm-5.3-flash'): (
        'correct', 'ONE call (remap_colormap, source_cmap=inferno), came back verified. '
                  'Looks almost unchanged by eye because the source corpus figure already '
                  'reads blue-ish, so this was checked by pixel diff against the original: '
                  '7.0% of pixels changed (310,068 px), matching the tool\'s own report '
                  'exactly, and both panels moved (top 3.6%, bottom 10.5%) -- the remap '
                  'reached the whole figure, not just one panel',
        'matches -- said it would re-theme the gradient and check the fit line; it did, '
        'in one efficient call'),
    ('chart_scatter', 'z-ai/glm-5.3-flash'): (
        'correct', 'points and colorbar both moved to the brand gradient in one call, '
                  'position-along-scale preserved',
        'matches -- said exactly this'),
    ('legend_bleed', 'z-ai/glm-5.3-flash'): (
        'correct', 'legend is opaque, swatches and text intact; the green and red curves '
                  'that used to show faintly through the box are now fully hidden behind '
                  'it -- which is correct, not a defect: an opaque legend rendered normally '
                  'would hide whatever sits behind it too, and the box was read tight to '
                  'the legend rather than oversized',
        'matches -- said the bleed-through is gone and the swatches/text survived, which '
        'is true'),
    ('lighthouse', 'z-ai/glm-5.3-flash'): (
        'correct', 'a clean crop to the tower, the foreground boardwalk gone -- the '
                  'cheap whole-image sanity check passed with no segmentation involved',
        'matches'),
    ('two_bicycles', 'z-ai/glm-5.3-flash'): (
        'correct', 'the bicycle keeps its colour, the ivy/brick background and a second, '
                  'barely-visible bicycle at the right edge both go black-and-white; a '
                  'small colour speck the mask caught got erased and inpainted cleanly',
        'matches'),
    ('door', 'z-ai/glm-5.3-flash'): (
        'correct', 'the CORE request is met -- door is a dark-ish green, postbox and '
                  'house-number plate are untouched red -- but the driver then went '
                  'further than asked (restoring the frosted-glass diamond panes and '
                  'chrome handle to their original look, which the request never '
                  'mentioned) and could not land that extra work: two attempts to fix the '
                  'handle each broke something else and were undone, leaving the panes '
                  'green-tinted and the handle blended into the door. Judged correct '
                  'because the thing actually asked for -- door green, postbox red -- is '
                  'delivered cleanly; the unfinished hardware detail is scope the driver '
                  'imposed on itself, not scope it dropped',
        'matches as far as it goes -- unusually transparent in real time about each '
        'failed sub-fix (reports the grey-door mistake, the bad generative_fill, both '
        'undos) but the transcript simply stops after "Undo:" with no closing summary, so '
        'a reader has to infer from the trail that the hardware work was abandoned'),
    ('two_dogs', 'z-ai/glm-5.3-flash'): (
        'wrong', 'the adult dog\'s entire head is now a FLAT solid tan blob -- no fur '
                'texture, no eye, no visible ear detail, just a flat-filled silhouette -- '
                'while its body stayed the original darker reddish-brown, so the two '
                'halves of the same animal do not even match each other, let alone the '
                'puppy. The driver was still mid-repair when the transcript ends '
                '("undo and instead try oklab-based recolour, which remaps lightness more '
                'fully") -- a real edit, visibly wrong, so wrong rather than incomplete',
        'MISLEADING BY OMISSION -- ends mid-plan with no acknowledgement that the '
        'delivered head is a flat, textureless patch'),
    ('two_cars_multi', 'z-ai/glm-5.3-flash'): (
        'wrong', 'neither car is cleanly blue -- the front Mazda keeps a large red-orange '
                 'patch across its front bumper/wheel arch and the rear Kia has a big red '
                 'patch on its side. "Both red cars blue" is not what shipped',
        'MISLEADING BY OMISSION -- transcript ends mid-cleanup ("let me preview a tight '
        'colour mask to see if it can cleanly target just the leftover car-red") with no '
        'statement that the delivered picture still has visible red patches on both cars'),
    ('two_cars', 'z-ai/glm-5.3-flash'): (
        'wrong', 'THE SHARPEST CONTRAST IN THIS RUN: same image, same request as the '
                'original-run\'s honest stop, but this driver decided unilaterally that '
                '"the front one" is "the red car" -- with no distinguishing feature in '
                'the request to support that choice -- and both cars ended up damaged: '
                'the front Mazda fully blue, the rear Kia ALSO mostly blue with ugly '
                'red-orange blotches where the recolour missed. The original run\'s '
                'correct behaviour on this exact frame (refuse, edit nothing) is the '
                'measurement this case exists to take, and a weaker/cheaper model did '
                'not take it',
        'the self-report is honest about the ambiguity ("since there are two red cars, my '
        'plan is: front Mazda = the red car") but treats picking one as a legitimate '
        'resolution rather than naming the ambiguity to the user and stopping -- the '
        'failure is a judgement call stated plainly, not an omission'),
    ('powerlines', 'z-ai/glm-5.3-flash'): (
        'incomplete', 'picture is UNEDITED -- one remove_object polyline attempt missed '
                     'the wires entirely ("wires are still fully present"), was undone, '
                     'and every remaining call was a zoom trying to pin down the wires\' '
                     'exact pixel path; it never attempted a second edit and never reached '
                     'the honest-stop conclusion this refusal-kind case was built to test. '
                     'Re-run at max_turns=20 to see whether more room produces a working '
                     'trace, a correct refusal, or the same unresolved search',
        'matches as far as it goes -- narrates the failed attempt and the ongoing search '
        'accurately, but the transcript simply stops mid-search with no summary'),
    ('bird_wire', 'z-ai/glm-5.3-flash'): (
        'correct', 'RE-RUN at max_turns=20 (the original 10-turn attempt abandoned after '
                  'only 2 calls with 8 turns to spare -- unexplained, see the session '
                  'notes; ruled out as the same max_tokens mechanism as annotation_box by '
                  'arithmetic, since its total completion tokens, 5,754, could not have '
                  'hit a 16,000 ceiling on any turn. That first retry attempt was also '
                  'OOM-killed once by the box\'s own memory pressure, unrelated to the '
                  'model). This clean run: erase_object came back verified=False (a '
                  'ghost), the driver undid it, used generative_fill as the documented '
                  'fallback (worked), then a second pass with remove_object on a small '
                  'leftover tail-streamer remnant (verified=True). The wire runs through '
                  'cleanly with its natural twist, sky is uninterrupted; one faint '
                  'hairline mark survives near the bottom edge but does not read as '
                  'bird-shaped',
        'matches -- the driver flagged the same faint remnant itself before finishing'),
    ('white_car', 'z-ai/glm-5.3-flash'): (
        'correct', 'RE-RUN at max_turns=20 after the original 10-turn attempt abandoned '
                  'mid-plan after only 3 calls with turns to spare (see the session notes '
                  '-- that first attempt used only 6,848 of a possible much larger token '
                  'budget, nowhere near the 16,000/turn ceiling, so it was NOT a '
                  'max_tokens truncation, just an unexplained early stop). This clean '
                  're-run used isolate_object correctly in 3 calls: car keeps its colour '
                  '(body, mirror, red side-marker, amber indicators), wall/ground/'
                  'vegetation go grey, verified ~85% of the car retained colour vs ~1% '
                  'outside. Same case, same model, different outcome on a second try -- '
                  'the abandonment looks stochastic, not deterministic on this input',
        'matches'),
    ('annotation_box', 'z-ai/glm-5.3-flash'): (
        'incomplete', 'ZERO tool calls and ZERO closing text -- usage shows '
                     'completion_tokens=16000, exactly the max_tokens ceiling, so this '
                     'looks like the model generating a long, unproductive response on '
                     'its very first turn and being cut off before it ever framed a tool '
                     'call or a sentence. CONFIRMED at the 20-turn re-run: IDENTICAL '
                     'result -- 0 calls, completion_tokens=16000 again, "(no closing '
                     'statement)" again. Extra turns cannot help a case that never gets '
                     'past turn 1; this is the predicted outcome of a genuinely different '
                     'mechanism from the turn-budget cases, not a re-run that needed to '
                     'be tried and wasn\'t -- see the handoff on agent.py\'s loop not '
                     'distinguishing a max_tokens truncation from a considered stop',
        '(nothing to compare -- driver_said is empty both times)'),
    ('cones', 'z-ai/glm-5.3-flash'): (
        'incomplete', 'STILL UNEDITED after the max_turns=20 re-run -- twelve calls '
                     'again, all sample_colour/preview_colour_mask/zoom, hunting for a '
                     'cone-colour sample tight enough to exclude the traffic lights and '
                     'fuel-pump panels the first mask also caught; isolate_colour/'
                     'replace_colour/recolour_object were never called on either attempt. '
                     'This is now a GENUINE result, not a turn-budget artifact: given five '
                     'times the room, this driver could not get past sampling a clean '
                     'cone colour on this photograph',
        'matches as far as it goes -- accurately narrates the contamination problem it '
        'found, but the transcript stops mid-search with no summary both times'),
}
