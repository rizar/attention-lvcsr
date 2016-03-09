def global_push_initialization_config(brick, initialization_config,
                                      filter_type=object):
    #TODO: this needs proper selectors! NOW!
    if not brick.initialization_config_pushed:
        raise Exception("Please push_initializatio_config first to prevent it "
                        "form overriding the changes made by "
                        "global_push_initialization_config")
    if isinstance(brick, filter_type):
        for k,v in initialization_config.items():
            if hasattr(brick, k):
                setattr(brick, k, v)
    for c in brick.children:
        global_push_initialization_config(
            c, initialization_config, filter_type)


def rename(var, name):
    var.name = name
    return var
