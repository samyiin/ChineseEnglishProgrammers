# pip install pandas torch transformers scikit-learn numpy
# pip install captum
from captum.attr import LayerIntegratedGradients
import torch


class BertTokenImportance:
    def __init__(self, model, embedding_layer, embed_to_token_aggregation_func="L1"):
        """
        This is a wrapper for Captum, since token space is not differentiable,
        so we can only get integrated gradients until the embedding layer

        I cannot support MPS because mps cannot have float64, so the device is either cpu or cuda
        Aggregation function is either L1 or L2 norm.
        :param model:
        :param embedding_layer:
        :param embed_to_token_aggregation_func:
        """
        # this can only be L1 or L2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_to_token_aggregation_func = embed_to_token_aggregation_func
        self.embedding_layer = embedding_layer.to(self.device)
        self.model = model.to(self.device)
        self.model.eval()

    def embedding_vector_importance(self, input_ids, attention_mask, baseline_input_ids, important_to_which_target_id):
        """

        :param important_to_which_target_id: 0 is the first output logit of model, 1 is the second logit
        :param input_ids:
        :param attention_mask:
        :param baseline_input_ids:
        :return:
        """

        def custom_forward(input_ids, attention_mask):
            """
            Because the model's output is SequenceClassifierOutput object,
            LayerIntegratedGradients need the forward function to return just logit
            we can either pass in the .forward() function or customize it
            :param input_ids:
            :param attention_mask:
            :return:
            """
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits

        lig = LayerIntegratedGradients(custom_forward, self.embedding_layer)

        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline_input_ids,
            additional_forward_args=(attention_mask,),
            target=important_to_which_target_id,
            return_convergence_delta=True
        )

        return attributions

    def token_importance(self, batch_of_one_sample, important_to_which_target_id=0):
        """
        We assume that Bert model's tokenized datasets have this two fields
        input_ids and attention_mask
        and everything is in tensor format.

        :param important_to_which_target_id: 0 is the first output logit of model, 1 is the second logit
        :param batch_of_one_sample: a batch of tokenized dataset with batch_size=1
        :return: attribution, shape size as the input ids
        """
        input_ids = batch_of_one_sample['input_ids'].to(self.device)
        attention_mask = batch_of_one_sample["attention_mask"].to(self.device)
        baseline_input_ids = torch.zeros_like(input_ids).to(self.device)

        attributions = self.embedding_vector_importance(input_ids, attention_mask, baseline_input_ids,
                                                        important_to_which_target_id)
        if self.embed_to_token_aggregation_func == "L1":
            token_attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        else:
            # todo: I haven't implement L2 norm yet, but we will probably not use L2 norm aggregation
            token_attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        # make sure all attribution are between 1 and -1
        token_attributions = token_attributions / max(abs(token_attributions))
        return token_attributions

    def word_pieces_importance(self, word_pieces, batch_of_one_sample, important_to_which_target_id=0):
        """
        helpful tool to map word pieces to importance,
        :param word_pieces: must be the same shape as the input_ids in batch_of_one_sample
        :param batch_of_one_sample:
        :param important_to_which_target_id:
        :return:
        """
        attention_mask = batch_of_one_sample["attention_mask"][0]
        token_attributions = self.token_importance(batch_of_one_sample,
                                                   important_to_which_target_id=important_to_which_target_id)
        filtered_word_pieces = [piece for piece, mask in zip(word_pieces, attention_mask) if mask == 1]
        filtered_token_attributions = [attr for attr, mask in zip(token_attributions, attention_mask) if mask == 1]
        unique_word_pieces = [f"({i}) {word}" for i, word in enumerate(filtered_word_pieces)]
        return unique_word_pieces, filtered_token_attributions
