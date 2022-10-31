# sound


## 1. DNS 생성 코드 
> sound source로 부터 이미지 형태의 DNS 생성 코드   
> [파일 경로]/result_image/ 에 생성된 dns 저장

```python 
###################################################################
[gate@dev-infra soundGateway]$ python3 dns/snd_dns_cal_new.py [파일 경로] [파일 이름]
```


## 2. wav to flac code

> wave 파일을 flac으로 변경하는 코드. 동일 위치에 [파일이름].flac 파일 생성됨   

```python
###################################################################
## 사용법 : -p 파일 경로 -f 파일 이름 
## help
[gate@dev-infra soundGateway]$ python3 util/wavtoflac.py --h
usage: wavtoflac.py [-h] [-p PATH] [-f FILENAME]

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  A wav file path to convert
  -f FILENAME, --fileName FILENAME
                        A fileName to convert

###################################################################
## 성공 시 return value
[gate@dev-infra soundGateway]$ python3 ./util/wavtoflac.py -p /gate/gitRepo/script/data/test -f wav_test.wav
True

###################################################################
## 실패 시 return 내용
[gate@dev-infra soundGateway]$ python3 ./util/wavtoflac.py -p /gate/gitRepo/script/data/test -f wav_test.wav
Error Exception

```

## 3. log. 

> util의 paiplog를 import 후 적용 시 ./logs_dir 에 에러 내용 logging

```
from logs import paiplog

@paiplog
def functionName() : 
....
```