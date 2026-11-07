Hook bir **düğümün toplam gradyanını** değiştirir, sonra o değer girdilere dağılır. $a$'nın iki yolu var: $d=c+a$ doğrudan (hook'tan etkilenmez) ve $c=ab$ üzerinden (etkilenir).

Saklanan tensör sorusunda `W` bir parametre; yine de matmul onu saklar (referans olarak: kopya değil, ama sayılan bellek o tensörün kendisi).
