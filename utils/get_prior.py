import numpy as np

def get_prior(shape, aspect_ratios, min_size, max_size, input_size):
    box_widths = []
    box_heights = [] 
    for ar in aspect_ratios:
        if ar == 1.0:
            box_widths.append(min_size)
            box_heights.append(min_size)
            if max_size:
                box_widths.append(np.sqrt(min_size * max_size))
                box_heights.append(np.sqrt(min_size * max_size))
        else:
            box_widths.append(min_size * np.sqrt(ar))
            box_heights.append(min_size / np.sqrt(ar))

    box_widths = 0.5 * np.array(box_widths)
    box_heights = 0.5 * np.array(box_heights)

    step_x = input_size / shape[1]
    step_y = input_size / shape[0]
    linx = np.linspace(0.5 * step_x, input_size - 0.5 * step_x, shape[1])
    liny = np.linspace(0.5 * step_y, input_size - 0.5 * step_x, shape[0])
    centers_x, centers_y = np.meshgrid(linx, liny)
    centers_x = np.reshape(centers_x, (-1, 1))
    centers_y = np.reshape(centers_y, (-1, 1))
    num_priors = len(box_widths)
    prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
    prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors))
    prior_boxes[:, ::4] -= box_widths
    prior_boxes[:, 1::4] -= box_heights
    prior_boxes[:, 2::4] += box_widths
    prior_boxes[:, 3::4] += box_heights
    prior_boxes[:, ::2] /= input_size
    prior_boxes[:, 1::2] /= input_size
    prior_boxes = np.reshape(prior_boxes, (-1, 4))
    prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
    return prior_boxes

def get_priors(input_size):
    prior1 = get_prior([40, 40], [1, 2, 1/2], min_size=30, max_size=None, input_size=input_size)
    prior2 = get_prior([20, 20], [1, 2, 3, 1/2, 1/3], min_size=60, max_size=114, input_size=input_size)
    prior3 = get_prior([10, 10], [1, 2, 3, 1/2, 1/3], min_size=114, max_size=168, input_size=input_size)
    prior4 = get_prior([5, 5], [1, 2, 3, 1/2, 1/3], min_size=168, max_size=222, input_size=input_size)
    prior5 = get_prior([3, 3], [1, 2, 3, 1/2, 1/3], min_size=222, max_size=276, input_size=input_size)
    prior6 = get_prior([1, 1], [1, 2, 3, 1/2, 1/3], min_size=276, max_size=330, input_size=input_size)
    priors = np.concatenate([prior1, prior2, prior3, prior4, prior5, prior6], axis=0)
    return priors
