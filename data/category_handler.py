
import re
# categories are split into four files. See api call below
# to get a list of strings ["", ""...""]

# some interesting things to note. there are a large number
# of fields which are labeled allocation flag, this means that
# the data was missing and was inferred from the dataset using one
# of several allocation methods (i.e. closest match on other parameters)
# with the flag signaling which allocation method
# 
# 
# also note that line number appears in multiple attributes,
# and remains a mystery as to whether it belongs in categorical
# or continuous, thus i have put them in unsure for now
# 
# also note that there is a short list of irrelevant,
# which i have made the choice of doing so because
# they were either id, date, or some other unique factor
# to each respondent


def get_attribute_list(name):
    categoriestest = open(name, "r")
    lines = categoriestest.read().split(',')
    lines = [re.sub(r"[\n\t\s]*", "", i) for i in lines]
    return lines

# call verify to verify the aggregation matches the original
# baseline
def verify():
    newlist = []
    # the four categories. Call them to get each category
    newlist.extend(get_attribute_list("cat_continuous.dat"))
    newlist.extend(get_attribute_list("cat_categorical.dat"))
    newlist.extend(get_attribute_list("cat_unsure.dat"))
    newlist.extend(get_attribute_list("cat_irrelevant.dat"))

    newlist.sort()


    baseline = get_attribute_list("categoriesbaseline.dat")
    baseline.sort()
    
    if (len(baseline) != len(newlist)):
        print("missing terms")

    for i in baseline:
        if i not in newlist:
            print("ERROR: newlist missing: " + str(i))
    for i in newlist:
        if i not in baseline:
            print("ERROR: newlist misspell: " + str(i))

    print("if no error messages, all category headers are verified")



verify()