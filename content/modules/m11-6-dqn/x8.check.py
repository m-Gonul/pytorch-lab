from labcheck import need, close, ensure


def check(ns,R):
    import torch
    def run():
        f=need(ns,'residual_step');expect=need(ns,'expected_split')
        w=torch.tensor([.5,.75,.25],dtype=torch.float64);x=torch.tensor([1.,0.,0.],dtype=torch.float64);b=torch.tensor([0.,1.,0.],dtype=torch.float64);c=torch.tensor([0.,0.,1.],dtype=torch.float64)
        close(f(w,x,b,0.,alpha=.1),[.525,.725,.25],what='tek örnek tam gradyanı')
        close(f(w,x,b,0.,c,alpha=.1),[.525,.75,.225],what='bağımsız ikinci örnek yönü')
        close(w,[.5,.75,.25],what='girdi ağırlıkları korunur')
        close(expect(w,False,'naive'),[0.,0.,0.],what='TDE optimumu')
        close(expect(w,False,'double'),[0.,.0625,-.0625],what='BE gradyan yönü farklı')
        close(expect(w,True,'double'),expect(w,True,'naive'),what='deterministik alttaki geçişler')
        ensure(not torch.equal(expect(w,False,'double'),expect(w,True,'double')),'aynı özellik verisi farklı BE yönü')
        R.table('Aynı gözlenen w, farklı koşullama',['model/yöntem','beklenen yön'],[['split/naive',str(expect(w,False,'naive').tolist())],['split/double',str(expect(w,False,'double').tolist())],['presplit/double',str(expect(w,True,'double').tolist())]],note='Sabit dışsal w için tüm olasılıklar sayıldı; gerçek gizli duruma reset yetkisi oracle bilgisidir.')
    R.test('Residual PyTorch gradyanı ve aynı durumdan iki örnek',run)
