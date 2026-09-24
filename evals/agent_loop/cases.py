"""Twenty requests, in the words someone would use, on images this package has never seen.

Images come from Wikimedia Commons into `evals/_corpus/agent_loop/` (credits in
evals/CREDITS.md), except the `chart_*` / `legend_bleed` / `annotation_box` figures,
which are synthetic and committed as data.

What each case tests lives in `kinds.py`, deliberately not here, so a driver only ever
gets the request and the image, the way a user types it.
"""
CASES = [
    {'id': 'door', 'image': 'evals/_corpus/agent_loop/door.jpg', 'request': 'Make the front door dark green. The postbox next to it must stay red.'},
    {'id': 'cat_cutout', 'image': 'evals/_corpus/agent_loop/cat.jpg', 'request': 'Cut the cat out onto a transparent background.'},
    {'id': 'taxi', 'image': 'evals/_corpus/agent_loop/taxi.jpg', 'request': 'Keep the yellow taxis in colour and turn the rest of the street grey.'},
    {'id': 'cones', 'image': 'evals/_corpus/agent_loop/roadworks.jpg', 'request': 'Make the traffic cones blue.'},
    {'id': 'two_cars', 'image': 'evals/_corpus/agent_loop/two_cars.jpg', 'request': 'Make the red car blue.'},
    {'id': 'heatmap', 'image': 'evals/_corpus/agent_loop/heatmap.png', 'request': 'Re-theme this heatmap to our brand blues, #00407A to #52BDEC, and keep '
                'what the colours mean.'},
    {'id': 'zipf', 'image': 'evals/_corpus/agent_loop/zipf.png', 'request': 'This figure uses default matplotlib colours. Put it in our brand palette: '
                'dark blue #00407A and light blue #52BDEC.'},
    {'id': 'cat_erase', 'image': 'evals/_corpus/agent_loop/cat.jpg', 'request': 'Remove the cat from the picture.'},

    {'id': 'bird_wire', 'image': 'evals/_corpus/agent_loop/bird.jpg', 'request': 'Remove the bird from the wire, please.'},
    {'id': 'powerlines', 'image': 'evals/_corpus/agent_loop/powerlines.jpg', 'request': 'Erase the power lines from this sky.'},
    {'id': 'lamppost', 'image': 'evals/_corpus/agent_loop/lamppost.jpg', 'request': 'Keep the street lamp in colour, turn the rest of the picture grey.'},
    {'id': 'lighthouse', 'image': 'evals/_corpus/agent_loop/lighthouse.jpg', 'request': 'Crop this down to just the lighthouse tower, cut out the '
                'boardwalk in front of it.'},
    {'id': 'white_car', 'image': 'evals/_corpus/agent_loop/white_car.jpg', 'request': 'Keep the white car in colour, turn everything else grey.'},
    {'id': 'two_dogs', 'image': 'evals/_corpus/agent_loop/two_dogs.jpg', 'request': 'Make both dogs the same golden colour.'},
    {'id': 'two_bicycles', 'image': 'evals/_corpus/agent_loop/two_bicycles.jpg', 'request': 'Keep the bicycle in colour, make the rest of the picture '
                'black and white.'},
    {'id': 'two_cars_multi', 'image': 'evals/_corpus/agent_loop/two_cars.jpg', 'request': 'Make both red cars blue.'},
    {'id': 'chart_bar', 'image': 'evals/_corpus/agent_loop/chart_bar.png', 'request': 'This bar chart uses the default matplotlib colours. Put it in our '
                'brand palette: dark blue #00407A and light blue #52BDEC.'},
    {'id': 'chart_scatter', 'image': 'evals/_corpus/agent_loop/chart_scatter.png', 'request': 'Re-theme this scatter plot to our brand blues, #00407A to '
                '#52BDEC, and keep the colorbar matching the points.'},
    {'id': 'legend_bleed', 'image': 'evals/_corpus/agent_loop/legend_bleed.png', 'request': 'The legend box in the middle of this chart is see-through '
                'and the lines behind it show through. Clean it up.'},
    {'id': 'annotation_box', 'image': 'evals/_corpus/agent_loop/annotation_box.png', 'request': "There's a text box covering part of the curve. Get rid "
                'of the box but keep the label and its arrow.'},
]
