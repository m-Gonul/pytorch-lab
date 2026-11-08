<!-- çözüm -->

Dört noktayı kareli kâğıda çiz. Başlangıç merkezleri kötü seçilmiş: ikisi de sol alttaki gruba yakın. İlk iterasyonda $c_2$ üç noktayı alıp sağa kayıyor, ikinci iterasyonda $(1,0)$ $c_1$'e geri dönüyor ve algoritma doğru iki kümeyi buluyor.

Bu sefer şanslıydık. $c_1=(0,0)$, $c_2=(4,4)$ ve beşinci bir nokta $(5,4)$ yerine $(10,10)$ olsaydı bir merkez aykırı değere kilitlenebilirdi; k-means++ ve çoklu başlatma bu yüzden var.
