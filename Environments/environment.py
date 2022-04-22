import random
import pickle

class Environment:
    # 此处修改 64 -》 100 -》64
    def __init__(self, env_id='FB15k', dim_state=64):
        self.env_id = env_id
        self.dim_state = dim_state
        self.ent_dict = dict()
        self.rel_dict = dict()
        self.num_ents = 0
        self.num_rels = 0
        self.h2r = dict()
        # 存储所有 头实体*10000+关系 -》 尾实体序列
        self.hr2t = dict()
        self.h2rt = dict()
        self.ht2r = dict()
        self.ht_set = set()
        self.tri_set = set()
        # 头实体*10000+关系 的list
        self.pair_set = set()
        self.model = None
        self.matrix = None
        self.cur_state = None
        self.p_r = 1
        self.p_e = 1


    def load(self):
        # load knowledge graph file
        triples = list()
        f = open('../../Algorithms/DQN/param_ent64_rel64_TransE.pkl', 'rb')
        data2 = pickle.load(f)
        entlist = data2.get('entlist')
        self.num_ents=len(entlist)
        self.ent_dict = dict(zip(entlist, range(self.num_ents)))
        # FB15K -> SVKG
        with open('../../Environments/KGs/{}/newTrain.txt'.format(self.env_id), 'r',
                  encoding='utf-8') as f:
        # with open('../../Environments/KGs/{}/freebase_mtr100_mte100-train.txt'.format(self.env_id), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                # FB15K -> SVKG
                h, r, t = line.rstrip('\n').split('\t')
                try:
                    self.rel_dict[r]
                except KeyError:
                    self.rel_dict[r] = self.num_rels
                    self.num_rels += 1
                h_id, r_id, t_id = self.ent_dict[h], self.rel_dict[r], self.ent_dict[t]
                triples.append([h_id, r_id, t_id])
                # if len(triples) > 2000:
                #     break
        print('Env_id={}, num_state={}, num_action={}'.format(self.env_id, self.num_ents, self.num_rels))
        # adjacency construction
        while self.p_r <= self.num_rels:
            self.p_r *= 10
        while self.p_e <= self.num_ents:
            self.p_e *= 10
        for tri in triples:
            h_id, r_id, t_id = tri[0], tri[1], tri[2]
            try:
                self.h2r[h_id].append(r_id)
            except KeyError:
                self.h2r[h_id] = [r_id]
            try:
                self.hr2t[h_id * self.p_r + r_id].append(t_id)
            except KeyError:
                self.hr2t[h_id * self.p_r + r_id] = [t_id]
            try:
                self.h2rt[h_id].append([r_id, t_id])
            except KeyError:
                self.h2rt[h_id] = [[r_id, t_id]]
            try:
                self.ht2r[h_id * self.p_r + t_id].append(r_id)
            except KeyError:
                self.ht2r[h_id * self.p_r + t_id] = [r_id]
            self.tri_set.add(h_id * self.p_r * self.p_e + r_id * self.p_e + t_id)
            self.pair_set.add(h_id * self.p_r + r_id)
            # KR ADD
            self.ht_set.add(h_id * self.p_r + t_id)


    # def load(self):
    #     # load knowledge graph file
    #     triples = list()
    #     # FB15K -> SVKG
    #     with open('../../Environments/KGs/{}/newTrain.txt'.format(self.env_id), 'r',
    #               encoding='utf-8') as f:
    #     # with open('../../Environments/KGs/{}/freebase_mtr100_mte100-train.txt'.format(self.env_id), 'r', encoding='utf-8') as f:
    #         for line in f.readlines():
    #             # FB15K -> SVKG
    #             h, r, t = line.rstrip('\n').split('\t')
    #             try:
    #                 self.ent_dict[h]
    #             except KeyError:
    #                 self.ent_dict[h] = self.num_ents
    #                 self.num_ents += 1
    #             try:
    #                 self.ent_dict[t]
    #             except KeyError:
    #                 self.ent_dict[t] = self.num_ents
    #                 self.num_ents += 1
    #             try:
    #                 self.rel_dict[r]
    #             except KeyError:
    #                 self.rel_dict[r] = self.num_rels
    #                 self.num_rels += 1
    #             h_id, r_id, t_id = self.ent_dict[h], self.rel_dict[r], self.ent_dict[t]
    #             triples.append([h_id, r_id, t_id])
    #             # if len(triples) > 2000:
    #             #     break
    #     print('Env_id={}, num_state={}, num_action={}'.format(self.env_id, self.num_ents, self.num_rels))
    #     # adjacency construction
    #     while self.p_r <= self.num_rels:
    #         self.p_r *= 10
    #     while self.p_e <= self.num_ents:
    #         self.p_e *= 10
    #     for tri in triples:
    #         h_id, r_id, t_id = tri[0], tri[1], tri[2]
    #         try:
    #             self.h2r[h_id].append(r_id)
    #         except KeyError:
    #             self.h2r[h_id] = [r_id]
    #         try:
    #             self.hr2t[h_id * self.p_r + r_id].append(t_id)
    #         except KeyError:
    #             self.hr2t[h_id * self.p_r + r_id] = [t_id]
    #         try:
    #             self.h2rt[h_id].append([r_id, t_id])
    #         except KeyError:
    #             self.h2rt[h_id] = [[r_id, t_id]]
    #         try:
    #             self.ht2r[h_id * self.p_r + t_id].append(r_id)
    #         except KeyError:
    #             self.ht2r[h_id * self.p_r + t_id] = [r_id]
    #         self.tri_set.add(h_id * self.p_r * self.p_e + r_id * self.p_e + t_id)
    #         self.pair_set.add(h_id * self.p_r + r_id)
    #         # KR ADD
    #         self.ht_set.add(h_id * self.p_r + t_id)


    # 取大小为2000的子图
    def load_child_gragh(self):
        rels_set =set()
        h2r_child = dict()
        # 存储所有 头实体*10000+关系 -》 尾实体序列
        hr2t_child = dict()
        h2rt_child = dict()
        ht_set_child = set()
        tri_set_child = set()
        # 头实体*10000+关系 的list
        pair_set_child = set()
        memory=list()
        first = random.randint(1, 1000)
        num, point = 0, 1
        tail = first
        while len(memory) < 2000:
            if tail not in memory:
                memory.append(tail)
                # print(num)
                num += 1
            head = tail
            try:
                r_t = random.sample(self.h2rt[head], 1)[0]
                h_id, r_id, tail = head, r_t[0], r_t[1]
                while h_id * self.p_r * self.p_e + r_id * self.p_e + tail in tri_set_child:
                    if len(self.h2rt[head]) == len(h2rt_child[head]):
                        raise KeyError("尾实体全部拿到")
                    r_t=random.sample(self.h2rt[head],1)[0]
                    r_id, tail = r_t[0], r_t[1]
                rels_set.add(r_id)
                try:
                    h2r_child[h_id].append(r_id)
                except KeyError:
                    h2r_child[h_id] = [r_id]
                try:
                    hr2t_child[h_id * self.p_r + r_id].append(tail)
                except KeyError:
                    hr2t_child[h_id * self.p_r + r_id] = [tail]
                try:
                    h2rt_child[h_id].append([r_id, tail])
                except KeyError:
                    h2rt_child[h_id] = [[r_id, tail]]

                tri_set_child.add(h_id * self.p_r * self.p_e + r_id * self.p_e + tail)
                pair_set_child.add(h_id * self.p_r + r_id)
                # KR ADD
                ht_set_child.add(h_id * self.p_r + tail)
            except KeyError:
                tail=memory[point]
                point += 1
        # print(memory)
        self.num_ents = 2000
        self.num_rels = len(rels_set)
        self.h2r = h2r_child
        # 存储所有 头实体*10000+关系 -》 尾实体序列
        self.hr2t = hr2t_child
        self.h2rt = h2rt_child
        self.ht_set = ht_set_child
        self.tri_set = tri_set_child
        # 头实体*10000+关系 的list
        self.pair_set = pair_set_child



    def seed(self, n):
        random.seed(n)

    def reset(self):
        self.cur_state = random.randint(0, self.num_ents - 1)
        return self.cur_state

    def cal_reward(self, state, action):
        # h_id = state
        # r_id = action
        # # 动作正确
        # if (h_id * self.p_r + r_id) in self.pair_set:
        #     reward = 10
        #     done = False
        # #未选中并且到达了终止节点
        # elif (action == self.num_rels) and (h_id * self.p_r + r_id not in (self.hr2t.keys())):
        #     reward = 10
        #     done = True
        # #选择错误
        # else:
        #     reward = -1
        #     done = True
        # return reward, done
        h_id = state
        t_id = action
        # 动作正确
        # print('h_id={},t_id={},h_id * self.p_r + t_id={}'.format(state,action,h_id * self.p_r + t_id))
        if (h_id * self.p_r + t_id) in self.ht_set:
            reward = 10
            done = False
        # 未选中并且到达了终止节点
        elif (action == self.num_ents):
            reward = 10
            done = True
        # 选择错误
        else:
            reward = -1
            done = True
        return reward, done

    def step(self, action):
        reward, done = self.cal_reward(self.cur_state, action)
        self.cur_state = action
        return self.cur_state, reward, done

    def action_sample(self):
        # try:
        #     sam = random.sample(self.h2rt[self.cur_state], 1)[0]
        #     r_id = sam[0]
        #     t_id = sam[1]
        #     action = r_id * self.num_ents + t_id
        #     return action
        # except KeyError:
        #     action = self.num_ents * self.num_rels
        #     return action
        action = random.randint(0, self.num_rels)
        return action

    def supervised_action(self, h):
        try:
            r = random.sample(self.h2r[h], 1)[0]
        except KeyError:
            r = self.num_rels
        return r


