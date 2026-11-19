// Own literal on-policy differential SARSA runs, source Example10.2.
// g++ -O3 -std=c++17 measure_ch10_access.cpp -o /tmp/access
// /tmp/access > measurements_ch10_access.json
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace std;
struct RNG{uint32_t a;double next(){uint32_t t=(a+=0x6d2b79f5u);t=(t^(t>>15))*(t|1u);t^=t+(t^(t>>7))*(t|61u);return double(t^(t>>14))/4294967296.;}};
template<class T>void arr(const vector<T>&v){cout<<'[';for(size_t i=0;i<v.size();i++){if(i)cout<<',';cout<<v[i];}cout<<']';}
int main(){cout<<setprecision(15);cout<<"{\"parameters\":{\"servers\":10,\"p\":0.06,\"alpha\":0.01,\"beta\":0.01,\"epsilon\":0.1,\"steps\":2000000,\"initial_Q\":0,\"initial_rho\":0,\"rng\":\"Mulberry32\"},\"cases\":[";bool comma=false;
 for(int include:{1,0})for(int seed:{1,2,1000001,1000002}){
  RNG r{uint32_t(seed)};vector<double>q(88);vector<long>visits(44);double rho=0,total=0,block=0,last=0;vector<double>gain,average,moving;vector<int>points;vector<array<double,10>>fixture;
  auto act=[&](int f,int p){if(!f)return 0;if(r.next()<.1)return int(r.next()*2);int s=(f*4+p)*2;if(q[s]==q[s+1])return int(r.next()*2);return int(q[s+1]>q[s]);};
  int f=10,p=int(r.next()*4),a=act(f,p);
  for(int t=0;t<2000000;t++){
   double reward=a*(1<<p);int after=f-a,busy=10-(include?after:f),nf=after;
   for(int j=0;j<busy;j++)nf+=r.next()<.06;
   int np=int(r.next()*4),na=act(nf,np);double delta=reward-rho+q[(nf*4+np)*2+na]-q[(f*4+p)*2+a];
   rho+=.01*delta;q[(f*4+p)*2+a]+=.01*delta;visits[f*4+p]++;total+=reward;block+=reward;if(t>=1900000)last+=reward;
   if(t<100)fixture.push_back({double(f),double(p),double(a),reward,double(nf),double(np),double(na),delta,rho,q[(f*4+p)*2+a]});
   f=nf;p=np;a=na;
   if((t+1)%10000==0){points.push_back(t+1);gain.push_back(rho);average.push_back(total/(t+1));moving.push_back(block/10000);block=0;}
  }
  if(comma)cout<<',';comma=true;cout<<"{\"release_new\":"<<(include?"true":"false")<<",\"seed\":"<<seed<<",\"steps\":2000000,\"rho\":"<<rho<<",\"reward_mean\":"<<total/2000000<<",\"last100000_mean\":"<<last/100000<<",\"Q\":";arr(q);cout<<",\"visits\":";arr(visits);cout<<",\"points\":";arr(points);cout<<",\"rho_curve\":";arr(gain);cout<<",\"reward_curve\":";arr(average);cout<<",\"block_curve\":";arr(moving);cout<<",\"first100\":[";
  for(size_t j=0;j<fixture.size();j++){if(j)cout<<',';cout<<'[';for(int k=0;k<10;k++){if(k)cout<<',';cout<<fixture[j][k];}cout<<']';}cout<<"]}";
 }
 cout<<"]}\n";
}
