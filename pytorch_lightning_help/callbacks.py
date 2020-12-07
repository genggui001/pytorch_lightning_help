import pytorch_lightning as pl

# 专门用于在on_train_epoch_end的时候计算分数


class EvalCallback(pl.callbacks.Callback):

    def __init__(self):
        super(EvalCallback, self).__init__()
        self.best_score = 0

    # def val_operation(self, trainer, train_model, outputs):
    #     val_y_pre = predict(val_x, trainer, train_model.cnnnet, batch_size=8)
    #     now_score, other_info = evaluation(val_true_y, val_y_pre)

    #     train_model.logger.experiment.add_scalar("val_acc", now_score, train_model.current_epoch)
    #     train_model.logger.experiment.add_text("val_label_score", other_info, train_model.current_epoch)

    #     return now_score

    def val_operation(self, trainer, train_model, outputs):
        raise Exception("虚方法需要重写")

    # def test_operation(self, trainer, train_model, outputs):
    #     test_y_pre = predict(test_x, trainer, train_model.cnnnet, batch_size=8)
    #     now_score, other_info = evaluation(test_true_y, test_y_pre)

    #     train_model.logger.experiment.add_scalar("test_acc", now_score, train_model.current_epoch)
    #     train_model.logger.experiment.add_text("test_label_score", other_info, train_model.current_epoch)

    #     return now_score

    def test_operation(self, trainer, train_model, outputs):
        raise Exception("虚方法需要重写")

    def on_train_epoch_end(self, trainer, train_model, outputs):

        # val 计算
        val_score = self.val_operation(trainer, train_model, outputs)
        # 加入val_score分数作为指标
        if val_score is not None:
            trainer.logger_connector.callback_metrics['val_score'] = val_score

        if val_score > self.best_score:
            self.best_score = val_score
            # 加入检验下的最佳分数
            train_model.logger.experiment.add_scalar("val_best_score", self.best_score, train_model.current_epoch)

            test_score = self.test_operation(trainer, train_model, outputs)
            # 加入test_score分数作为指标
            if test_score is not None:
                trainer.logger_connector.callback_metrics['test_score'] = test_score

        return None


# 专门用于在on_train_epoch_end的时候保存参数


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, monitor='val_score', save_top_k=1, save_last=True, **kargs):
        super(ModelCheckpoint, self).__init__(monitor=monitor,
                                              save_top_k=save_top_k, save_last=save_last, **kargs)

    def on_validation_end(self, trainer, pl_module):
        pass

    def on_train_epoch_end(self, trainer, train_model, outputs):
        super(ModelCheckpoint, self).save_checkpoint(trainer, train_model)

# 在反复测试的时候也能使用同一个进度条


class ProgressBar(pl.callbacks.ProgressBar):

    def __init__(self):
        super().__init__()
        self.my_test_bar = None

    def init_test_tqdm(self):
        if self.my_test_bar is None:
            self.my_test_bar = pl.callbacks.progress.tqdm(
                desc='Testing',
                position=(2 * self.process_position + 1),
                disable=self.is_disabled,
                leave=True,
                dynamic_ncols=True,
                file=pl.callbacks.progress.sys.stdout
            )

        self.my_test_bar.reset(
            pl.callbacks.progress.convert_inf(self.total_test_batches))
        return self.my_test_bar

    def on_test_end(self, trainer, pl_module):
        super(pl.callbacks.ProgressBar, self).on_test_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        if self.my_test_bar is not None:
            self.my_test_bar.close()
