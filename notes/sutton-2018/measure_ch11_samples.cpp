// Own shared behavior streams; source Baird diagram, no source curves copied.
// g++ -O3 -std=c++17 measure_ch11_samples.cpp -o /tmp/ch11 && /tmp/ch11 > measurements_ch11_samples.json
#include <array>
#include <vector>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;
struct RNG{uint32_t a;double next(){uint32_t t=(a+=0x6d2b79f5u);t=(t^(t>>15))*(t|1u);t^=t+(t^(t>>7))*(t|61u);return double(t^(t>>14))/4294967296.;}};
using W=array<double,8>;
W phi(int s){W x{};if(s<6){x[s]=2;x[7]=1;}else{x[6]=1;x[7]=2;}return x;}
double dot(const W&a,const W&b){double z=0;for(int i=0;i<8;i++)z+=a[i]*b[i];return z;}
W initial(){return W{1,1,1,1,1,1,10,1};}
double norm(const W&w){return sqrt(dot(w,w));}
double rms(const W&w){double a=0;for(int s=0;s<7;s++){double v=dot(w,phi(s));a+=v*v/7;}return sqrt(a);}
void arr(const W&w){cout<<'[';for(int i=0;i<8;i++){if(i)cout<<',';cout<<w[i];}cout<<']';}
struct Learner{W w=initial(),v{};double follow=0;bool failed=false;int steps=0;};
void update(Learner&l,int alg,int s,int ns,bool solid,double prevRho){if(l.failed)return;const double rho=solid?7:0,g=.99,alpha=alg==0?.01:alg==3?.03:.005,beta=.05;W x=phi(s),xp=phi(ns);double delta=g*dot(l.w,xp)-dot(l.w,x),vx=dot(l.v,x);l.follow=1+g*prevRho*l.follow;
 W dw{},dv{};for(int i=0;i<8;i++){if(alg==0)dw[i]=alpha*rho*delta*x[i];else if(alg==1)dw[i]=alpha*rho*(delta*x[i]-g*xp[i]*vx);else if(alg==2)dw[i]=alpha*rho*(x[i]-g*xp[i])*vx;else dw[i]=alpha*l.follow*rho*delta*x[i];if(alg==1||alg==2)dv[i]=beta*rho*(delta-vx)*x[i];}
 for(int i=0;i<8;i++){l.w[i]+=dw[i];l.v[i]+=dv[i];if(!isfinite(l.w[i])||abs(l.w[i])>1e100)l.failed=true;}l.steps++;
}
int main(){cout<<setprecision(15);cout<<"{\"shared_stream_steps\":400000,\"source_sample_steps\":1000,\"Q_steps\":100000,\"samples\":[";bool comma=false;
 for(int seed:{1,2,1000001,1000002}){
  RNG rng{uint32_t(seed)};int s=int(rng.next()*7);double prevRho=0;array<Learner,4>L{};array<W,2>Q{initial(),initial()};vector<array<double,13>>hist;vector<array<double,10>>qh;vector<array<double,5>>fixture;array<int,14>visits{};
  for(int t=0;t<=100000;t++){
   if(t<=1000&&t%5==0){array<double,13>z{};z[0]=t;for(int k=0;k<4;k++){z[1+3*k]=rms(L[k].w);z[2+3*k]=norm(L[k].w);z[3+3*k]=L[k].follow;}hist.push_back(z);}
   if(t%500==0){array<double,10>z{};z[0]=t;z[1]=sqrt(dot(Q[0],Q[0])+dot(Q[1],Q[1]));for(int i=0;i<8;i++)z[2+i]=Q[1][i];qh.push_back(z);}
   if(t==100000)break;
   bool solid=rng.next()>=6./7;int ns=solid?6:int(rng.next()*6);int a=solid?1:0;visits[s*2+a]++;
   if(t<1000)for(int k=0;k<4;k++)update(L[k],k,s,ns,solid,prevRho);
   double target=.99*max(dot(Q[0],phi(ns)),dot(Q[1],phi(ns))),d=target-dot(Q[a],phi(s));W x=phi(s);for(int i=0;i<8;i++)Q[a][i]+=.01*d*x[i];
   if(t<100)fixture.push_back({double(s),double(a),double(ns),d,sqrt(dot(Q[0],Q[0])+dot(Q[1],Q[1]))});
   s=ns;prevRho=solid?7:0;
  }
  if(comma)cout<<',';comma=true;cout<<"{\"seed\":"<<seed<<",\"sample_history\":[";
  for(size_t j=0;j<hist.size();j++){if(j)cout<<',';cout<<'[';for(int k=0;k<13;k++){if(k)cout<<',';cout<<hist[j][k];}cout<<']';}cout<<"],\"final\":[";
  for(int k=0;k<4;k++){if(k)cout<<',';cout<<"{\"mode\":"<<k<<",\"steps\":"<<L[k].steps<<",\"censored\":"<<(L[k].failed?"true":"false")<<",\"weights\":";arr(L[k].w);cout<<",\"aux\":";arr(L[k].v);cout<<'}';}
  cout<<"],\"Q_weights\":[";arr(Q[0]);cout<<',';arr(Q[1]);cout<<"],\"Q_visits\":[";for(int k=0;k<14;k++){if(k)cout<<',';cout<<visits[k];}cout<<"],\"Q_history\":[";
  for(size_t j=0;j<qh.size();j++){if(j)cout<<',';cout<<'[';for(int k=0;k<10;k++){if(k)cout<<',';cout<<qh[j][k];}cout<<']';}cout<<"],\"first100\":[";
  for(size_t j=0;j<fixture.size();j++){if(j)cout<<',';cout<<'[';for(int k=0;k<5;k++){if(k)cout<<',';cout<<fixture[j][k];}cout<<']';}cout<<"]}";
 }
 cout<<"],\"ETD_distribution\":{\"runs\":4096,\"steps_each\":1000,\"seeds\":\"1..4096, selected in advance\",\"RMSVE\":[";
 vector<int>cuts;double maxTrace=0;long actualDist=0;
 for(int seed=1;seed<=4096;seed++){RNG r{uint32_t(seed)};int s=int(r.next()*7);Learner l;double rho=0;for(int t=0;t<1000;t++){bool solid=r.next()>=6./7;int ns=solid?6:int(r.next()*6);update(l,3,s,ns,solid,rho);maxTrace=max(maxTrace,l.follow);rho=solid?7:0;s=ns;if(l.failed)break;}actualDist+=l.steps;if(seed>1)cout<<',';if(l.failed){cout<<"null";cuts.push_back(seed);}else cout<<rms(l.w);}
 cout<<"],\"censored_seeds\":[";for(size_t i=0;i<cuts.size();i++){if(i)cout<<',';cout<<cuts[i];}cout<<"],\"actual_steps\":"<<actualDist<<",\"complete_runs\":"<<4096-cuts.size()<<",\"max_follow\":"<<maxTrace<<"}}\n";
}
