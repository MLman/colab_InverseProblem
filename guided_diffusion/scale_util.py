

def make_scale_dict(exp_name):
    """
    Scale dict for VGG_Gram exp
    """
    
    scale_dict = {'content':None, 'style':None}

    if 'contentT' in exp_name:
        scale_dict['content'] = 'time'
    elif 'contentR' in exp_name:
        scale_dict['content'] = 'reversed'
    elif 'contentR' in exp_name:
        scale_dict['content'] = 'exp'
    elif 'contentN' in exp_name:
        scale_dict['content'] = 'None'
    else:
        raise NotImplementedError

    if 'styleT' in exp_name:
        scale_dict['style'] = 'time'
    elif 'styleR' in exp_name:
        scale_dict['style'] = 'reversed'
    elif 'styleR' in exp_name:
        scale_dict['style'] = 'exp'
    elif 'styleN' in exp_name:
        scale_dict['style'] = 'None'
    else:
        raise NotImplementedError
    
    return scale_dict

def get_scale(scale_type, scale_dict, norm_loss):

    content_type = scale_type['content']
    style_type = scale_type['style']

    content_scale = scale_dict[content_type] * norm_loss
    style_scale = scale_dict[style_type] * norm_loss

    return content_scale, style_scale