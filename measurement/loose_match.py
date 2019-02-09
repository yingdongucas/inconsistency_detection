
def loose_compare_version_point(cve_id, db, version_point_set_1, version_point_set_2):
    if version_point_set_1 == set() or version_point_set_2 == set():
        return True
    version_1_in_version_2 = version_set_1_in_version_set_2(version_point_set_1, version_point_set_2)
    version_2_in_version_1 = version_set_1_in_version_set_2(version_point_set_2, version_point_set_1)
    version_same = version_1_in_version_2 or version_2_in_version_1

    # CVE-2004-0079 securityfocus_official not loosely matched
    # {'6.2', '6', '6.3', '6.1'}
    # {'4480', '4490'}
    if is_base_diff(version_point_set_1, version_point_set_2):
        return True

    match_flg, direction = False, ''
    if version_1_in_version_2 and not version_2_in_version_1:
        direction = 'under'
        match_flg = True
    elif not version_1_in_version_2 and version_2_in_version_1:
        direction = 'over'
        match_flg = True

    elif not version_1_in_version_2 and not version_2_in_version_1:
        if debug_mode:
            print(cve_id, db, 'not loosely matched')
            print(version_point_set_1)
            print(version_point_set_2)

    else:
        # todo: can't find versions in dictionary
        return True

    if debug_mode and not version_same:
            print()
            print(version_point_set_1)
            print(version_point_set_2)
            print()

    if debug_mode and direction in ['under', 'over']:
        print(cve_id, db, direction)
        print(version_point_set_1)
        print(version_point_set_2)
        print('========================')

    return match_flg, direction


def version_set_1_in_version_set_2(version_point_set_1, version_point_set_2):
    # version_point_set_1 should be standard/nvd version
    for version_1 in version_point_set_1:
        if version_1 not in version_point_set_2:
            return False
    return True


def is_base_diff(version_point_set_1, version_point_set_2):
    base_set_1 = get_version_base(version_point_set_1)
    base_set_2 = get_version_base(version_point_set_2)
    diff = 100
    for base_1 in base_set_1:
        for base_2 in base_set_2:
            if abs(base_1 - base_2) >= diff:
                return True


def get_version_base(version_point_set):
    base_set = set()
    for version in version_point_set:
        base = version.split('.')[0]
        if base.isdigit():
            base_set.add(int(base))
    return base_set


