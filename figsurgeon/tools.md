# What each operation does, and what a person says to ask for it

This file is the catalogue the plain-language router reads (`figsurgeon/intent.py`). It is
not documentation *about* the code — it is an input *to* it, so adding an operation means
writing its entry here, and nothing in the router needs to change.

Each entry has three parts, and each earns its place:

- **Does** — one line, written as the OUTCOME. Several of these operations take the same
  kind of request in the same words ("cut the sign out" is either an erase or an extract),
  and what separates them is which of the two things survives, the thing or the scene.
- **Needs** — the arguments that cannot be guessed. An operation whose needs cannot be
  filled from the sentence is refused rather than called with a hole in it.
- **Sounds like** — example requests, in the words a person would actually use. These are
  what the sentence is compared against. Write them the way someone who has never read this
  package would phrase it, and do NOT copy them from the test battery: the battery is the
  held-out measure, and an example lifted from it measures nothing.

---

## recolour_object
**Does:** the named thing changes colour; everything else stays as it was.
**Needs:** subject, to_rgb
**Sounds like:**
- make the sofa green
- paint the door red
- change the roof to grey
- i want the jacket in navy
- turn the tractor yellow
- her coat should be burgundy instead

## erase_object
**Does:** the named thing goes away and the background is filled in behind it; the rest of the scene stays.
**Needs:** subject
**Sounds like:**
- delete the rubbish bin
- take the cyclist out
- lose the cable in the corner
- can you remove that bollard
- get the tourist out of the frame
- paint out the road sign

## isolate_object
**Does:** the named thing keeps its colour and everything else is desaturated; nothing is removed.
**Needs:** subject
**Sounds like:**
- leave the bicycle in colour and desaturate the rest
- only the flowers should stay coloured
- grey everything except the boat
- colour pop the child
- keep the umbrella coloured, mute everything else

## extract_object
**Does:** the named thing survives alone on transparency; the scene goes.
**Needs:** subject
**Sounds like:**
- cut the mug out on its own
- give me the sign as a transparent png
- isolate the bottle onto transparency
- export just the badge with alpha
- lift the shoe off the background

## scale_object
**Does:** the named thing changes size in place, about its own centre; the rest of the scene stays the same size.
**Needs:** a box or point on the object, and a scale factor
**Sounds like:**
- make the moon smaller without touching anything else
- shrink the logo a bit but leave the rest of the image alone
- the star should be twice as big, same position
- grow the icon by half its current size

## generative_fill
**Does:** a boxed region is redrawn from scratch by a hosted image model, to invent content nothing else here can construct; everything outside the box is untouched.
**Needs:** a box, and what the region should contain
**Sounds like:**
- make up some texture for that empty square so it doesn't look blank
- invent something plausible in that patch, it doesn't need to be real
- extend the pattern into that missing square, guessing what continues there
- there's nothing usable in that region, generate something that could belong there

## remap_colormap
**Does:** a chart's colour scale becomes a different one while each pixel keeps the value it encoded; the colorbar and every panel go with it.
**Needs:** new_colours (source_cmap if the caller names it)
**Sounds like:**
- change the colour scale of this heatmap to our palette
- this uses inferno, re-theme it to company colours
- swap the rainbow scale for a two colour gradient
- recolour the colour ramp and its bar
- restyle the magma colormap as blues

## clean_transparent_box
**Does:** a see-through legend or annotation box stops showing the data behind it, while its own swatches and text stay.
**Needs:** nothing the caller must type; a box helps
**Sounds like:**
- the key is translucent and the plot shows through it, tidy that
- clean the bleed through behind the legend
- remove the ghost of the data inside the legend panel
- the annotation box is see through, fix it
- wipe what shows through the label box

## replace_font
**Does:** the chart's text is redrawn in another typeface at the same size and position.
**Needs:** a font
**Sounds like:**
- set the tick labels in helvetica
- change the chart typeface to our font
- redo the axis text in a different font
- restyle the labels with another typeface

## isolate_colour
**Does:** pixels near one colour keep it and the rest of the frame is desaturated. Colour-driven, not object-driven.
**Needs:** target_rgb
**Sounds like:**
- keep the yellow and drop the rest of the colour
- everything mono except the green bits
- colour pop the orange
- only the blue stays
- desaturate all but the red

## replace_colour
**Does:** every pixel near one colour becomes another, anywhere in the frame.
**Needs:** from_rgb, to_rgb
**Sounds like:**
- change every red pixel to blue
- make the yellow bits pink instead
- recolour the green areas as brown
- the purple should read as teal

## sample_colour
**Does:** reports the RGB of a region; changes nothing.
**Needs:** a region
**Sounds like:**
- what rgb is that area
- read the colour off that patch
- tell me the exact shade there
- pick the colour from that spot

## blur_background
**Does:** everything except the subject is blurred; the subject stays sharp.
**Needs:** nothing
**Sounds like:**
- soften what is behind the subject
- give it a shallow depth of field look
- defocus the backdrop
- make the background less sharp

## remove_background
**Does:** the subject survives on transparency; the background goes.
**Needs:** nothing
**Sounds like:**
- knock out the backdrop
- make the background transparent
- strip everything behind the subject
- give me it without a background

## replace_background
**Does:** the subject is composited onto a new plain background.
**Needs:** colour_rgb
**Sounds like:**
- drop the subject onto a plain backdrop
- put a solid colour behind her
- swap the backdrop for white
- new background, plain grey

## adjust
**Does:** brightness, contrast, saturation or sharpness of the whole frame move up or down.
**Needs:** nothing
**Sounds like:**
- lighten it
- more punch please
- make it less flat
- crank the colours
- it looks dull
- too bright, pull it back

## stylise
**Does:** a whole-frame look — greyscale, sepia, or a darkened edge.
**Needs:** nothing
**Sounds like:**
- monochrome it
- make it look old and brown
- shade the edges down
- go greyscale
- add a dark edge around it
- drain the colour out of it

## denoise
**Does:** sensor noise and grain are smoothed while edges are kept.
**Needs:** nothing
**Sounds like:**
- it is speckly, smooth that out
- reduce the noise from the high iso
- get rid of the graininess

## auto_white_balance
**Does:** a colour cast is neutralised so whites read as white.
**Needs:** nothing
**Sounds like:**
- the whites look yellow, neutralise them
- correct the colour cast
- it has an orange tint, fix it

## crop
**Does:** the frame is cut down; what is outside the kept region is gone.
**Needs:** a region
**Sounds like:**
- trim the edges off
- cut it down to the middle section
- keep only the left portion
- chop off the bottom strip

## rotate
**Does:** the whole frame turns by an angle.
**Needs:** an angle
**Sounds like:**
- turn it a quarter clockwise
- it is on its side, spin it upright
- rotate by ninety

## flip
**Does:** the whole frame mirrors, left-right or top-bottom.
**Needs:** an axis
**Sounds like:**
- mirror it left to right
- flip it top to bottom
- reverse it horizontally

## resize
**Does:** the frame is scaled to a new size; nothing else changes.
**Needs:** a size
**Sounds like:**
- scale it down to a thousand pixels
- make it half the size
- shrink it for the web

## add_text
**Does:** a caption is drawn onto the image.
**Needs:** text
**Sounds like:**
- write a caption across the top
- stamp a word on it
- put a label in the corner

## show_grid
**Does:** labelled pixel coordinates are drawn over the image so a caller can name a region. Changes nothing.
**Needs:** nothing
**Sounds like:**
- overlay coordinates so i can point at things
- put a ruler on the picture
- i need pixel positions

## preview_colour_mask
**Does:** shows which pixels a colour selection would touch, without touching them.
**Needs:** target_rgb
**Sounds like:**
- which pixels would that colour select
- show the selection before you apply it
- preview what that would hit

## preview_object_mask
**Does:** shows the mask an object edit would use, without editing.
**Needs:** subject
**Sounds like:**
- show me the mask for that object first
- outline what you would edit there

## undo
**Does:** the last edit is taken back.
**Needs:** nothing
**Sounds like:**
- take that back
- revert the last thing
- that was wrong, step back

## reset
**Does:** every edit is discarded and the original returns.
**Needs:** nothing
**Sounds like:**
- go back to the original
- throw away all my edits
- start from scratch
