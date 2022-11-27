# FastSpeech2 Implementation

This is the homework for the course [Deep Learning for Audio](https://github.com/markovka17/dla) at the [CS Faculty](https://cs.hse.ru/en/)
  of [HSE](https://www.hse.ru/en/).

## Task
We were given the FastSpeech implementation and the goal was to implement FastSpeech2 (i. e. add voice pitch and energy control).

## Examples
You can find generated examples with target phrase 

`My name is Yulya and this is the result of my FastSpeech2 model.`

in `example` directory.

## Test Model
You may use `fastspeech2_test.ipynb` file to test the model. You need to download it and to run the cells one by one. Change test texts in the punultimate cell if needed.

## Trainig
Training pipeline is avaliable at `fastspeech2_train.ipynb`.

## Model Weights
You can download model weights with the following code:

```shell
!wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm= \
$(wget --quiet --save-cookies /tmp/cookies.txt \
--keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=11UH13WOhwAcagACaARpr_mnqIiQQrXCJ' \
-O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=11UH13WOhwAcagACaARpr_mnqIiQQrXCJ" \
-O checkpoint_117000.pth.tar \
&& rm -rf /tmp/cookies.txt
```

```python
mel_config = MelSpectrogramConfig()
model_config = FastSpeechConfig()
train_config = TrainConfig()

model = FastSpeech2(model_config, train_config, mel_config)
model = model.to(train_config.device)

model.load_state_dict(torch.load('./checkpoint_117000.pth.tar', map_location='cuda:0')['model'])
model = model.eval()
```
