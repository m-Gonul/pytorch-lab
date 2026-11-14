// Chapter 8 Figure 8.8 / Exercise 8.8. Actual expected backups, no fitted curves.
// Build: g++ -O3 -std=c++17 -pthread measure_ch8_trajectory.cpp -o /tmp/ch8trajectory
// Run: /tmp/ch8trajectory > measurements_ch8_trajectory.json
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>
using namespace std;
struct RNG {
 uint32_t a; explicit RNG(uint32_t seed):a(seed?seed:1){}
 double operator()(){uint32_t t=(a+=0x6d2b79f5u);t=(t^(t>>15))*(1u|t);t^=t+(t^(t>>7))*(61u|t);return double(t^(t>>14))/4294967296.;}
 int index(int n){return int((*this)()*n);}
 double normal(){return sqrt(-2*log(max((*this)(),1e-15)))*cos(6.2831853071795864769*(*this)());}
};
struct Model {
 int S,b; vector<int> next; vector<double> reward;
 Model(int s,int branches,int seed):S(s),b(branches),next(2*S*b),reward(2*S*b){
  RNG rng(seed);
  for(int k=0;k<2*S;k++)for(int j=0;j<b;j++){
   int z;bool repeat;do{z=rng.index(S);repeat=false;for(int h=0;h<j;h++)if(next[k*b+h]==z)repeat=true;}while(repeat);
   next[k*b+j]=z;reward[k*b+j]=rng.normal();
  }
 }
 double backup(const vector<double>& Q,int k)const{double v=0;for(int j=0;j<b;j++){int n=next[k*b+j];v+=reward[k*b+j]+max(Q[2*n],Q[2*n+1]);}return .9*v/b;}
 // V = r + .9 P_pi V. Fixed greedy tie convention: first action.
 double evaluate(const vector<double>& Q,double& residual)const{
  vector<int> policy(S);vector<double> r(S),v(S,0);
  for(int i=0;i<S;i++){int k=2*i+(Q[2*i+1]>Q[2*i]);policy[i]=k;for(int j=0;j<b;j++)r[i]+=.9*reward[k*b+j]/b;}
  if(b==1){
   vector<int> colour(S,0),position(S,-1),path;path.reserve(S);
   for(int root=0;root<S;root++)if(!colour[root]){
    path.clear();int s=root;
    while(!colour[s]){colour[s]=1;position[s]=path.size();path.push_back(s);s=next[policy[s]];}
    int prefix=path.size();
    if(colour[s]==1){
     int st=position[s];double total=0,pow=1;
     for(int h=st;h<int(path.size());h++){total+=pow*r[path[h]];pow*=.9;}
     v[s]=total/(1-pow);double after=v[s];
     for(int h=int(path.size())-1;h>st;h--){v[path[h]]=r[path[h]]+.9*after;after=v[path[h]];}
     prefix=st;
    }
    for(int h=prefix-1;h>=0;h--){int i=path[h];v[i]=r[i]+.9*v[next[policy[i]]];}
    for(int i:path)colour[i]=2;
   }
  }else{
   // Subtract the start-state equation. This removes the slow constant eigenmode.
   vector<double>w(S,0),u(S,0);double v0=0;
   for(int iter=0;iter<10000;iter++){
    double base=r[0];for(int j=0;j<b;j++)base+=.9*w[next[policy[0]*b+j]]/b;
    double diff=0;
    for(int i=0;i<S;i++){double x=r[i];for(int j=0;j<b;j++)x+=.9*w[next[policy[i]*b+j]]/b;u[i]=x-base;diff=max(diff,abs(u[i]-w[i]));}
    w.swap(u);v0=base/.1;
    if(diff<1e-12)break;
   }
   double base=r[0];for(int j=0;j<b;j++)base+=.9*w[next[policy[0]*b+j]]/b;v0=base/.1;
   for(int i=0;i<S;i++)v[i]=w[i]+v0;
  }
  residual=0;
  for(int i=0;i<S;i++){double x=r[i];for(int j=0;j<b;j++)x+=.9*v[next[policy[i]*b+j]]/b;residual=max(residual,abs(v[i]-x));}
  return v[0];
 }
};
struct Result{vector<vector<double>> value;double residual=0;uint64_t backups=0;};
Result task(int S,int b,int steps,const vector<int>& points,int seed){
 Model model(S,b,seed);Result out;out.value=vector<vector<double>>(2,vector<double>(points.size()));
 for(int method=0;method<2;method++){
  vector<double> Q(2*S,0);RNG rng(seed+900000);int state=0,cp=0;
  for(int t=0;t<=steps;t++){
   if(cp<int(points.size())&&t==points[cp]){double res;out.value[method][cp++]=model.evaluate(Q,res);out.residual=max(out.residual,res);}
   if(t==steps)break;
   if(method==0){int k=t%(2*S);Q[k]=model.backup(Q,k);}
   else{
    int a;if(rng()<.1)a=rng.index(2);else if(Q[2*state]==Q[2*state+1])a=rng.index(2);else a=Q[2*state+1]>Q[2*state];
    int k=2*state+a;Q[k]=model.backup(Q,k);
    state=rng()<.1?0:model.next[k*b+rng.index(b)];
   }
   out.backups++;
  }
 }
 return out;
}
int main(int argc,char**argv){
 if(argc>1&&string(argv[1])=="verify"){
  cout<<setprecision(17)<<"[";
  for(int b:{1,3}){if(b==3)cout<<",";Model m(17,b,800001);auto out=task(17,b,100,{0,10,50,100},800001);
   cout<<"{\"S\":17,\"b\":"<<b<<",\"next\":[";for(int j=0;j<int(m.next.size());j++){if(j)cout<<",";cout<<m.next[j];}
   cout<<"],\"reward\":[";for(int j=0;j<int(m.reward.size());j++){if(j)cout<<",";cout<<m.reward[j];}
   cout<<"],\"values\":[";for(int method=0;method<2;method++){if(method)cout<<",";cout<<"[";for(int j=0;j<4;j++){if(j)cout<<",";cout<<out.value[method][j];}cout<<"]";}cout<<"]}";
  }cout<<"]\n";return 0;
 }
 int runs=argc>1?stoi(argv[1]):200;
 cout<<setprecision(15)<<"{\"runs\":"<<runs<<",\"seed_base\":800001,\"terminal_probability\":0.1,\"terminal_reward\":0,\"gamma\":1,\"nonterminal_reward\":\"iid N(0,1) for each branch\",\"successors\":\"distinct within each (s,a)\",\"cases\":[";
 vector<pair<int,int>> cases={{1000,1},{1000,3},{1000,10},{10000,1},{10000,3}};
 for(int c=0;c<int(cases.size());c++){
  auto[S,b]=cases[c];int steps=20*S;vector<int> points={0};
  // Fine initial sweep measurements expose the Exercise8.7 scallops.
  for(int t=S/10;t<=4*S;t+=S/10)points.push_back(t);
  for(int t=5*S;t<=steps;t+=S)points.push_back(t);
  vector<Result> results(runs);atomic<int> index{0};vector<thread> workers;
  for(int j=0;j<4;j++)workers.emplace_back([&](){int k;while((k=index++)<runs)results[k]=task(S,b,steps,points,800001+k);});
  for(auto& w:workers)w.join();
  double maxres=0;uint64_t total=0;for(auto&r:results){maxres=max(maxres,r.residual);total+=r.backups;}
  cerr<<"S="<<S<<" b="<<b<<" runs="<<runs<<" backups="<<total<<" residual="<<maxres<<"\n";
  if(maxres>1e-9)return 2;
  if(c)cout<<",";cout<<"{\"S\":"<<S<<",\"b\":"<<b<<",\"steps_per_method\":"<<steps<<",\"actual_backups\":"<<total<<",\"max_policy_Bellman_residual\":"<<maxres<<",\"points\":[";
  for(int j=0;j<int(points.size());j++){if(j)cout<<",";cout<<points[j];}cout<<"],\"mean\":[";
  for(int m=0;m<2;m++){if(m)cout<<",";cout<<"[";for(int j=0;j<int(points.size());j++){double avg=0;for(auto&r:results)avg+=r.value[m][j]/runs;if(j)cout<<",";cout<<avg;}cout<<"]";}cout<<"],\"se\":[";
  for(int m=0;m<2;m++){if(m)cout<<",";cout<<"[";for(int j=0;j<int(points.size());j++){double avg=0,ss=0;for(auto&r:results)avg+=r.value[m][j]/runs;for(auto&r:results)ss+=pow(r.value[m][j]-avg,2);if(j)cout<<",";cout<<(runs>1?sqrt(ss/(runs-1)/runs):0);}cout<<"]";}cout<<"]}";
 }
 cout<<"]}\n";
}
