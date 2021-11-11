import torch
import torch.nn as nn

from music_voice_classification.src.models.model import DEVICE
from music_voice_classification.src.models.model import multitemporalfeturemap
from music_voice_classification.src.utils.datasets import train_dl, valid_dl, test_dl


# model
model = multitemporalfeturemap(2, 2).to(DEVICE)

# model: load, save path name
model_genre_path = 'music_genre_01.pth'

# parameters
epochs = 3
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):

    torch.cuda.empty_cache()
    history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    best_val_loss = None
    for epoch in range(epochs):
        model.train()

        train_losses = []
        lrs = []

        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs

        model.epoch_end(epoch, result)
        history.append(result)

        # 검증 오차가 가장 적은 최적의 모델을 저장
        if not best_val_loss or result['val_loss'] < best_val_loss:
            torch.save(model.state_dict(), '../trained_models/' + model_genre_path)
            best_val_loss = result['val_loss']

    return history


history = [evaluate(model, valid_dl)]
print(history)


'''
    train
'''
history += train(epochs, max_lr, model, train_dl, valid_dl,
                 grad_clip=grad_clip,
                 weight_decay=weight_decay,
                 opt_func=opt_func)


'''
    test
'''
classes = ['fast', 'slow']

# for audio, label in test_dl:
#     print('classes: ', classes[label[0].item()])
#     break

model.load_state_dict(torch.load('../trained_models/' + model_genre_path))
model.eval()


def predict_audio(audio, model):
    # Convert to a batch of 1
    xb = audio
    xb = xb.to(DEVICE)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return classes[preds[0].item()]


print('test_dl size is ', len(test_dl))
true_count = 0
for audio, label in test_dl:
    print("Actual label: {}".format(classes[label[0].item()]))
    print('Predicted Label: ', predict_audio(audio, model))

    if classes[label[0].item()] == predict_audio(audio, model):
        true_count += 1
print("correct:", true_count)


# ===========================================
# ============ 시각적으로 보기 =================
# ===========================================
# def plot_accuracies(history):
#     accuracies = [x['val_acc'] for x in history]
#     plt.plot(accuracies, '-x')
#     plt.xlabel('epoch')
#     plt.ylabel('accuracy')
#     plt.title('Accuracy vs. No. of epochs')
# plot_accuracies(history)
#
# def plot_losses(history):
#     train_losses = [x.get('train_loss') for x in history]
#     val_losses = [x['val_loss'] for x in history]
#     plt.plot(train_losses, '-bx')
#     plt.plot(val_losses, '-rx')
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.legend(['Training', 'Validation'])
#     plt.title('Loss vs. No. of epochs')
# plot_losses(history)
#
# def plot_lrs(history):
#     lrs = np.concatenate([x.get('lrs', []) for x in history])
#     plt.plot(lrs)
#     plt.xlabel('Batch no.')
#     plt.ylabel('Learning rate')
#     plt.title('Learning Rate vs. Batch no.')
# plot_lrs(history)


