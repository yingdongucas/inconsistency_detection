import os
# pc_root_path = '/Users/yingdong/Desktop/project_data/'
# lrs_root_path = '/home/yzd57/vulnerability_data/'
# project_path = pc_root_path if os.path.exists('/Users/') else lrs_root_path

# need to modify:
root_path = '/home/yzd57/' if os.path.exists('/home/yzd57/') else '/Users/yingdong/Desktop/'

project_path = root_path + 'ying/data/'
data_write_path = root_path + 'ying/data/'

ner_data_suffix = '.txt_cut_199'

labeled_data_dir = 'ner_re_dataset'

labeled_ner_data_input_path = project_path + labeled_data_dir + '/ner_data_input/'
labeled_re_data_input_path = project_path + labeled_data_dir + '/re_data_input/'

# need to modify:

# ================= re npy path ====================

labeled_re_data_write_path = data_write_path + 're_npy_data/'
vec_npy_path_and_name = labeled_re_data_write_path + 'vec.npy'

word2id_file_name = 'word2id_file'
word2id_file_path_and_name = labeled_re_data_write_path + word2id_file_name + '.py'

id2word_file_name = 'id2word_file'
id2word_file_path_and_name = labeled_re_data_write_path + id2word_file_name + '.py'

char2id_file_name = 'char2id_file'
char2id_file_path_and_name = labeled_re_data_write_path + char2id_file_name + '.py'

relation2id_file_path_and_name = labeled_re_data_write_path + 'relation2id.txt'

# ================= re output path ====================

re_output_path = data_write_path + '/re_data_output/'
labeled_re_data_output_after_transfer_path = re_output_path + 'after_transfer/'
labeled_re_data_output_before_transfer_path = re_output_path + 'before_transfer/'

# ================= ner output path ====================

ner_output_path = data_write_path + '/ner_data_output/'
labeled_ner_data_output_after_transfer_path = ner_output_path + 'after_transfer/'
labeled_ner_data_output_before_transfer_path = ner_output_path + 'before_transfer/'

# ================= log output path ====================

sh_output_path = data_write_path + '/logs/'

# ================= model path ====================

model_path = data_write_path + 'models/'

ner_model_prefix = 'NER_model_'
ner_model_path = model_path + 'ner_model/'
ner_model_path_before_transfer = ner_model_path + 'before_transfer/'
ner_model_path_after_transfer = ner_model_path + 'after_transfer/'

re_model_prefix = 'RE_model_'
re_model_path = model_path + 're_model/'
re_model_path_before_transfer = re_model_path + 'before_transfer/'
re_model_path_after_transfer = re_model_path + 'after_transfer/'

# ================= dataset related ====================

software_label = 'S-vulnerable_software'
version_label = 'S-vulnerable_version'
labels = [software_label, version_label, 'O']
relation2id = {'y': 1, 'n': 0}
id2relation = {1: 'y', 0: 'n'}
cat_list = ['memc', 'bypass', 'csrf', 'dirtra', 'dos', 'execution', 'fileinc', 'gainpre', 'httprs', 'infor', 'overflow',
            'sqli', 'xss']
num_cat_dict = dict()
for cat in cat_list:
    num_cat_dict[len(num_cat_dict)] = cat

report_list = ['nvd', 'cve', 'securityfocus_official', 'securitytracker', 'securityfocus', 'edb', 'openwall']

# ================= word embeddings ====================

word_emb_dim_ner = 300
word_emb_dim_re = 50

# need to modify:
word_emb_path = root_path + 'ying/data/corpus_and_embeddings/embeddings/'

hash_file = word_emb_path + 'words' + str(word_emb_dim_ner) + '.lst'
emb_file = word_emb_path + 'embeddings' + str(word_emb_dim_ner) + '.txt'
word_emb_path_and_name = word_emb_path + 'word_emb' + str(word_emb_dim_re) + '.txt'

re_max_len = 50
re_max_len_char = 20

ner_max_len = 200
ner_max_len_char = 20






