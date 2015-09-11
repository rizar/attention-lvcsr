from blocks.model import Model


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


class SpeechModel(Model):
    def set_parameter_values(self, param_values):
        filtered_param_values = {
            key: value for key, value in param_values.items()
            # Shared variables are now saved separately, thanks to the
            # recent PRs by Dmitry Serdyuk and Bart. Unfortunately,
            # that applies to all shared variables, and not only to the
            # parameters. That's why temporarily we have to filter the
            # unnecessary ones. The filter deliberately does not take into
            # account for a few exotic ones, there will be a warning
            # with the list of the variables that were not matched with
            # model parameters.
            if not ('shared' in key
                    or 'None' in key)}
        super(SpeechModel,self).set_parameter_values(filtered_param_values)