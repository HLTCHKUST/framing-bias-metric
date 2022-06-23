
# normal version
def create_source_target_with_processing(objs, phase):
    
    target_path = 'data/acl2022_lrc_roundup_random_order/{}.target'.format(phase)
    source_path = 'data/acl2022_lrc_roundup_random_order/{}.source'.format(phase)

    for idx, obj in enumerate(objs):
        # center always in the beginning
        # left, right --> half/half

        left_body = remove_dotdotdot(" ".join(obj['left']['newBody']))
        right_body = remove_dotdotdot(" ".join(obj['right']['newBody']))
        center_body = remove_dotdotdot(" ".join(obj['center']['newBody']))
        
        # for equal order of political orientation
        
        all_bodies = [('L', left_body), ('R', right_body), ('C', center_body)]
        random.shuffle(all_bodies)


        source = " [SEP] ".join([item[1] for item in all_bodies]).replace("\n", "")
        source_order_string = " [SEP] ".join([item[0] for item in all_bodies])
        
        target = " ".join(obj['roundup']).replace("\n", "")
        

        with open(target_path, "a") as target_file:
            target_file.write(target)
            target_file.write("\n")

        with open(source_path, "a") as source_file: 
            source_file.write(source)
            source_file.write("\n")


        with open('data/acl2022_lrc_roundup_random_order.source_order.{}.txt'.format(phase), "a") as outfile:
            outfile.write(source_order_string)
            outfile.write("\n")

# for allsides
# split all_objs into train/val/test
article_train, article_not_train = train_test_split(PREPROCESSED_FILTERED_OBJ, test_size=0.2, random_state=42)
article_val, article_test = train_test_split(article_not_train, test_size=0.5, random_state=42)

create_source_target_with_processing(article_train, 'train')
create_source_target_with_processing(article_val, 'val')
create_source_target_with_processing(article_test, 'test')




# probe version
def create_source_target_with_processing_probe_format(objs, phase):
    
    target_path = 'data/acl2022_lrc_roundup_random_order_probe/{}.target'.format(phase)
    source_path = 'data/acl2022_lrc_roundup_random_order_probe/{}.source'.format(phase)

    for idx, obj in enumerate(objs):

        left_body = remove_dotdotdot(" ".join(obj['left']['newBody']))
        right_body = remove_dotdotdot(" ".join(obj['right']['newBody']))
        center_body = remove_dotdotdot(" ".join(obj['center']['newBody']))
        
        left_title = obj['left']['newsTitle']
        right_title = obj['right']['newsTitle']
        center_title = obj['center']['newsTitle']
        
        
        all_bodies = [('L', left_body, left_title), ('R', right_body, right_title), ('C', center_body, center_title)]
        random.shuffle(all_bodies)

        source = " [SEP] ".join([ "TITLE=> {}. ARTICLE=> {}".format(item[2],item[1]) for item in all_bodies]).replace("\n", "")
        source_order_string = " [SEP] ".join([item[0] for item in all_bodies])
        
        print("===Source===", source, "\n")
        
        roundup = " ".join(obj['roundup']).replace("\n", "")
        target = "TITLE=> {}. ARTICLE=> {}".format(obj['issue'], roundup)
            
            
#         # version 2
#         print("===Target===", target, "\n")
        
#         source = " [SEP] ".join([ "TITLE=> {}. ARTICLE=> {}".format(item[2],item[1]) for item in all_bodies]).replace("\n", "")
#         source_order_string = " [SEP] ".join([item[0] for item in all_bodies])
        
#         source += " TITLE=> {}. ARTICLE=>".format(obj['issue'])
        
#         roundup = " ".join(obj['roundup']).replace("\n", "")
#         target = roundup

        
        with open(target_path, "a") as target_file:
            target_file.write(target)
            target_file.write("\n")

        with open(source_path, "a") as source_file: 
            source_file.write(source)
            source_file.write("\n")


# for allsides
# split all_objs into train/val/test
article_train, article_not_train = train_test_split(PREPROCESSED_FILTERED_OBJ, test_size=0.2, random_state=42)
article_val, article_test = train_test_split(article_not_train, test_size=0.5, random_state=42)

create_source_target_with_processing_probe_format(article_train, 'train')
create_source_target_with_processing_probe_format(article_val, 'val')
create_source_target_with_processing_probe_format(article_test, 'test')