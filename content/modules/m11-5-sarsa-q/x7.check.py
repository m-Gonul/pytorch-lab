from labcheck import need


def check(ns, R_):
    def task():
        q_target = need(ns, 'q_target')
        learn_q_chain = need(ns, 'learn_q_chain')
        import math
        S=[0,1,2];pi=[[.75,.25]]*3;b=[[.5,.5]]*3;Q=[[0.,0.],[2.,4.],[0.,0.]]
        vals=[]
        for action in [0,1]:
            G,w=q_target(S,[0,action],[0.,0.],Q,pi,b,0,1,"is")
            vals.append(w*G)
        assert abs(sum(vals)/2-2.5)<1e-12  # The final sampled action's rho is essential.
        for method in ["expected","cv","tree","sigma"]:
            for sigma in [0,.5,1]:
                G,w=q_target(S,[0,0],[0.,0.],Q,pi,b,0,1,method,sigma)
                assert abs(G-2.5)<1e-12 and w==1
        # One real terminal step: no action A_T exists.
        for method in ["sarsa","is","expected","cv","tree","sigma"]:
            G,w=q_target([0,1],[0],[2.],[[0.,0.],[0.,0.]],pi,b,0,4,method)
            assert G==2. and w==1
        learned=learn_q_chain(2000,4,.05,"tree",0,0)
        assert len(learned)==5 and all(len(row)==2 for row in learned)
        assert learned[4]==[0.,0.] and abs(learned[3][0]-2)<1e-10 and learned[3][1]==0
        assert all(math.isfinite(x)for row in learned for x in row)
    R_.test('n-adım Q: oran/ufuk/terminal sınırları', task)
