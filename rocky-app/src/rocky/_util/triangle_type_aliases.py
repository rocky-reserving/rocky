## this is the logic to map possible misspellings of triangle types to the
## correct triangle type

# def get_aliases()-> dict:
        # allowed_triangle_types = [
        #     'paid_loss',
        #     'reported_loss',
        #     'paid_dcce',
        #     'paid_loss_dcce',
        #     'reported_loss_dcce',
        #     'case_reserves',
        #     'reported_counts',
        #     'closed_counts',
        #     'open_counts']

    # inital mapping of aliases to triangle types (same as allowed_triangle_types)
    # triangle_type_aliases = {'paid_loss': 'paid_loss','reported_loss': 'reported_loss','paid_dcce': 'paid_dcce','paid_loss_dcce': 'paid_loss_dcce','reported_loss_dcce': 'reported_loss_dcce','case_reserves': 'case_reserves','reported_counts': 'reported_counts','closed_counts': 'closed_counts','open_counts': 'open_counts'}

    # loop through the allowed triangle types and add aliases from common misspellings
    # for k in triangle_type_aliases.keys():
    #     triangle_type_aliases[k.replace("losses", "loss")] = k
    # triangle_type_aliases = add_alias(triangle_type_aliases, "_loss", "_loss_", "")
    # triangle_type_aliases = add_alias(triangle_type_aliases, "reported", replacement="rpt")
    # triangle_type_aliases = add_alias(triangle_type_aliases, "counts", replacement="count")
    # triangle_type_aliases = add_alias(triangle_type_aliases, "cnt", replacement="count")
    # triangle_type_aliases = add_alias(triangle_type_aliases, "resv", replacement="reserves")
    # triangle_type_aliases = add_alias(triangle_type_aliases, "rsv", replacement="reserves")
    
    # for k in triangle_type_aliases.keys():
    #     if k.contains("_loss") and k.find("_loss_") == -1:
    #         triangle_type_aliases[k.replace("_loss", "")] = k

    # for k in triangle_type_aliases.keys():
    #     if k.contains("reported"):
    #         triangle_type_aliases[k.replace("reported", "rpt")] = k
        
    # for k in triangle_type_aliases.keys():
    #     if k.contains("counts"):
    #         triangle_type_aliases[k.replace("counts", "count")] = k

    # for k in triangle_type_aliases.keys():
    #     if k.contains("cnt"):
    #         triangle_type_aliases[k.replace("cnt", "count")] = k

    # for k in triangle_type_aliases.keys():
    #     if k.contains("resv"):
    #         triangle_type_aliases[k.replace("resv", "reserves")] = k

    # for k in triangle_type_aliases.keys():
    #     if k.contains("rsv"):
    #         triangle_type_aliases[k.replace("rsv", "reserves")] = k

    # return triangle_type_aliases