from datetime import datetime

class Explanation():
    def __init__(self, id, methods, neural_network, parameters):
        self.id = id
        self.methods = methods
        self.neural_network = neural_network
        self.parameters = parameters
        self.date = datetime.now()
        self.results = {}

    def save_result(self, method, output_image, mask, filtered_image, elapsed_time, 
                    pred_top1, pred_top5, second_pass_pred, result_intersect, img_id, coco_masks, use_coco=False, coco_categories=None):
        
        if method not in self.results:
            self.results[method] = {}

        self.results[method][img_id] = {
            "output_image": output_image,
            "mask": mask,
            "filtered_image": filtered_image,
            "elapsed_time": elapsed_time,
            "pred_top1": pred_top1,
            "pred_top5": pred_top5,
            "second_pass": second_pass_pred,
            "result_intersect": result_intersect,
            "use_coco": use_coco,
            "coco_categories": coco_categories,
            "date": self.date,
            "coco_masks": coco_masks
        }



        