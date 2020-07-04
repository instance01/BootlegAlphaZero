Bad = B; Ok = K; Good = G

| Parameters   | TF Event file | Minutes | Result | Comments |
| ------------ | ------------- | ------- | ------ | -------- |
| 102 | `Jun21-12:12:31-284-pyrit.cip.ifi.lmu.de-mtcar-102` | | | |
| 103 | `Jun21-12:12:31-472-labradorit.cip.ifi.lmu.de-mtcar-103` | ~3945 | B: 10<br> K: 0<br> G: 0 | |
| 104 | `Jun21-12:12:31-495-smaragd.cip.ifi.lmu.de-mtcar-104` | | B: 10<br> K: 0<br> G: 0 | |
| 105 | `Jun21-12:12:31-527-lapislazuli.cip.ifi.lmu.de-mtcar-105` | | B: 10<br> K: 0<br> G: 0 | |
| 106 | `Jun21-12:12:31-627-tigerauge.cip.ifi.lmu.de-mtcar-106` | | B: 9<br> K: 1<br> G: 0 | |
| 107 | `Jun21-12:12:31-871-peridot.cip.ifi.lmu.de-mtcar-107` | | B: 9<br> K: 1<br> G: 0 | |
| 108 | `Jun21-12:12:31-735-petalit.cip.ifi.lmu.de-mtcar-108` | | B: 9<br> K: 1<br> G: 0 | |
| 109 | `Jun21-12:12:31-27-saphir.cip.ifi.lmu.de-mtcar-109` | | B: 9<br> K: 1<br> G: 0 | |
| 110 | `Jun21-12:12:31-477-thulit.cip.ifi.lmu.de-mtcar-111` | | B: 9<br> K: 1<br> G: 0 | Deeper net architecture makes things slower. |
| 111 | `Jun21-12:12:31-477-thulit.cip.ifi.lmu.de-mtcar-111` | | B: 10<br> K: 0<br> G: 0 | |
| 112 | `Jun21-12:12:31-617-tansanit.cip.ifi.lmu.de-mtcar-112` | | B: 6<br> K: 4<br> G: 0 | pb\_c\_init seems to have a good effect, yet again. |
| 113 | `Jun21-12:12:31-565-katzenauge.cip.ifi.lmu.de-mtcar-113` | | B: 7<br> K: 3<br> G: 0 | reduce\_eval scheduler with consecutive=true has no effect |
| 114 | `Jun21-12:12:31-66-sodalith.cip.ifi.lmu.de-mtcar-114` | | B: 8<br> K: 1<br> G: 1 | The one that was learnt so well was excellent. pb\_c\_init set to `0.1`, pb\_c\_base set to `1000`. |
| 115 | `Jun21-12:12:31-685-karneol.cip.ifi.lmu.de-mtcar-115` | | B: 10<br> K: 0<br> G: 0 | More exploration (dirichlet), pb\_c\_init set to `0.15`. |
| 116 | `Jun21-12:12:31-629-rubin.cip.ifi.lmu.de-mtcar-116` | | B: 9<br> K: 1<br> G: 0 | L2 (weight decay) |
| 117 | `Jun21-12:12:31-916-zirkon.cip.ifi.lmu.de-mtcar-117` | | B: 10<br> K: 0<br> G: 0 | pb\_c\_base set to 200, probably too low. |
| 118 | `Jun22-11:17:35-135-jaspis.cip.ifi.lmu.de-mtcar-118` | | B: 8<br> K: 0<br> G: 2 | reduce\_eval scheduler with consecutive=false |
| 119 | `Jun22-11:17:34-174-topas.cip.ifi.lmu.de-mtcar-119` | | B: 8<br> K: 0<br> G: 2 | reduce\_eval scheduler with consecutive=false |
| 120 | | | | Same as 123. Had issues with segfaults on pyrit. Really weird. |
| 121 | `Jun25-08:55:10-442-rubin.cip.ifi.lmu.de-mtcar-121` | | B: 10<br> K: 0<br> G: 0 | |
| 122 | `Jun25-07:41:48-869-peridot.cip.ifi.lmu.de-mtcar-122` | | B: 8<br> K: 0<br> G: 2 | |
| 123 | `Jun25-07:42:20-287-saphir.cip.ifi.lmu.de-mtcar-123` | | B: 5<br> K: 1<br> G: 4 | Really weird result. |
| 124 | `Jun25-18:32:20-691-rhodonit.cip.ifi.lmu.de-mtcar-124` | | B: 6<br> K: 2<br> G: 2 | |
| 125 | `Jun28-08:08:03-825-leubas.cip.ifi.lmu.de-mtcar-125` | | B: 8<br> K: 0<br> G: 2 | |
| 126 | `Jun28-08:08:03-445-jachen.cip.ifi.lmu.de-mtcar-126` | | B: 9<br> K: 0<br> G: 0 | |
| 127 | `Jun28-08:08:03-902-blau.cip.ifi.lmu.de-mtcar-127` | | B: 4<br> K: 3<br> G: 4 | pb\_c\_init set low, alpha halved. |
| 128 | `Jun28-08:08:03-762-rottach.cip.ifi.lmu.de-mtcar-128` | | B: 7<br> K: 0<br> G: 2 | |
