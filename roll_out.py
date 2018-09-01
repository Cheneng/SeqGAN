import torch
import torch.nn.functional as F
import copy


class Rollout(object):
    def __init__(self, model, update_rate):
        """
        :param model: the model to do the roll out.
        :param update_rate: update the parameter rate
            in other to easily sample the data a bit more random
        """
        self.ori_model = model
        self.rolling_model = copy.deepcopy(model)
        self.update_rate = update_rate

    def get_reward(self, data, num, discriminator):
        """
            To get the reward of every action of the sequence making.
        :param data: The input of the discriminator action. [batch_size, action_len]
        :param num:  The number of the sample times. (The larger the better but consuming more time and computation)
        :param discriminator:  The discriminator to compute the score.
        :return:
        """
        batch_size = data.size(0)
        seq_len = data.size(1)
        reward = []
        for i in range(num):
            for j in range(1, seq_len+1):
                temp_data = self.rolling_model.partial_sample(seq_len, data[:, :j])
                pred_reward = F.softmax(discriminator(temp_data), dim=1)  # tensor
                # If the first time to get the reward.
                if i == 0:
                    reward.append(pred_reward[:, 1].unsqueeze(1))
                else:
                    reward[j-1] += pred_reward[:, 1].unsqueeze(1)
        # the judge the whole sequence
        # whole_reward = F.softmax(discriminator(data), dim=1)
        reward = torch.cat(reward, dim=1) / num     # [ batch_size, seq_len ]
        # reward = torch.cat([reward, whole_reward], dim=1) # add the last whole reward
        return reward

    def update_param(self):
        """
        update the parameter with the the update_rate percent origin model.
        """
        for (name1, param1), (name2, param2) in zip(self.ori_model.parameter(), self.rolling_model):
            if name1 != name2:
                raise ValueError("The models parameter has been change")
            param1.data = self.update_rate * param1.data + (1 - self.update_rate) * param2.data

