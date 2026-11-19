#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>
using namespace std;
struct RNG{uint32_t a;double next(){uint32_t t=(a+=0x6d2b79f5u);t=(t^(t>>15))*(t|1u);t^=t+(t^(t>>7))*(t|61u);return double(t^(t>>14))/4294967296.;}};
constexpr int DIM=2640;
using F=array<int,8>;
struct State{double x,v;};
int div8(int q){return q>=0?q/8:-((-q+7)/8);}
F base_features(State s){int qx=int(floor(64*s.x/1.7)),qv=int(floor(64*s.v/.14));F f;for(int t=0;t<8;t++){int x=div8(qx+t),v=div8(qv+3*t);f[t]=((t*10+x+6)*11+v+4)*3;if(f[t]<0||f[t]+2>=DIM)abort();}return f;}
F action_features(const F&f,int a){F z=f;for(int&i:z)i+=a;return z;}
double value(const vector<double>&w,const F&f){double q=0;for(int i:f)q+=w[i];return q;}
int action(const vector<double>&w,const F&f,RNG&r){double q[3]={};for(int i:f)for(int a=0;a<3;a++)q[a]+=w[i+a];double m=max({q[0],q[1],q[2]});int ties[3],k=0;for(int a=0;a<3;a++)if(q[a]==m)ties[k++]=a;return ties[int(r.next()*k)];}
State step(State s,int a){s.v=clamp(s.v+.001*(a-1)-.0025*cos(3*s.x),-.07,.07);s.x=clamp(s.x+s.v,-1.2,.5);if(s.x<=-1.2)s.v=0;return s;}
struct Result{vector<double>lengths,w;long long steps=0;int failed=-1;};
Result learn(int n,double totalalpha,int episodes,int seed){Result o;vector<double>w(DIM);RNG start{uint32_t(seed)},act{uint32_t(seed+99173)};double alpha=totalalpha/8;const int INF=1000000000;
 for(int e=0;e<episodes;e++){State s{-.6+.2*start.next(),0};vector<F>fs;F f=base_features(s);fs.push_back(action_features(f,action(w,f,act)));int T=INF;
  for(int t=0;;t++){
   if(t<T){int a=fs[t][0]%3;s=step(s,a);o.steps++;if(s.x>=.5)T=t+1;else{f=base_features(s);fs.push_back(action_features(f,action(w,f,act)));}if(t>=1999999&&T==INF){o.failed=e;o.w=w;return o;}}
   int tau=t-n+1;
   if(tau>=0){int end=min(T,tau+n);double G=-(end-tau);if(tau+n<T)G+=value(w,fs[tau+n]);double d=alpha*(G-value(w,fs[tau]));if(!isfinite(d)||abs(d)>1e100){o.failed=e;o.w=w;return o;}for(int i:fs[tau])w[i]+=d;}
   if(tau==T-1){o.lengths.push_back(T);break;}
  }
 }
 o.w=w;return o;}
void arr(const vector<double>&v){cout<<'[';for(size_t i=0;i<v.size();i++){if(i)cout<<',';if(isfinite(v[i]))cout<<v[i];else cout<<"null";}cout<<']';}
struct Aggregate{vector<double>mean,se;long long steps=0;int complete=0;vector<int>failed;};
Aggregate combine(const vector<Result>&r,int ep){Aggregate o;o.mean.assign(ep,0);o.se.assign(ep,0);for(int j=0;j<int(r.size());j++){o.steps+=r[j].steps;if(r[j].failed<0&&int(r[j].lengths.size())==ep)o.complete++;else o.failed.push_back(j);}if(o.complete!=100){for(double&x:o.mean)x=numeric_limits<double>::quiet_NaN();return o;}for(int e=0;e<ep;e++){for(auto&x:r)o.mean[e]+=x.lengths[e]/100.;for(auto&x:r)o.se[e]+=pow(x.lengths[e]-o.mean[e],2);o.se[e]=sqrt(o.se[e]/9900.);}return o;}
vector<Result> trials(int n,double alpha,int ep){vector<Result>r(100);
#pragma omp parallel for schedule(dynamic)
 for(int j=0;j<100;j++)r[j]=learn(n,alpha,ep,1010001+j);return r;}
void printagg(const Aggregate&a){cout<<"{\"mean\":";arr(a.mean);cout<<",\"se\":";arr(a.se);cout<<",\"actual_steps\":"<<a.steps<<",\"complete_runs\":"<<a.complete<<",\"failed_runs\":[";for(size_t j=0;j<a.failed.size();j++){if(j)cout<<',';cout<<a.failed[j];}cout<<"]}";}
int main(){cout<<setprecision(14);vector<pair<int,double>>settings={{1,.1},{1,.2},{1,.5},{8,.3}};cout<<"{\"rng\":\"Mulberry32; separate start/action streams\",\"seeds\":[1010001,1010100],\"episodes\":500,\"runs\":100,\"full\":[";
 vector<double>fixture;
 for(size_t i=0;i<settings.size();i++){auto[n,a]=settings[i];auto results=trials(n,a,500);auto ag=combine(results,500);if(i)cout<<',';cout<<"{\"n\":"<<n<<",\"alpha_times8\":"<<a<<",\"results\":";printagg(ag);cout<<'}';cerr<<"full "<<n<<' '<<a<<" done"<<'\n';if(i==2)fixture.assign(results[0].lengths.begin(),results[0].lengths.begin()+min(size_t(50),results[0].lengths.size()));}
 cout<<"],\"early\":[";bool first=true;
 for(int n:{1,2,4,8,16}){double maxa=n==16?.6:n==8?.95:n==4?1.6:1.8;vector<double>alphas;for(double a=.1;a<=maxa+1e-9;a*=1.2)alphas.push_back(a);for(auto[n0,a]:settings)if(n0==n&&a<=maxa)alphas.push_back(a);sort(alphas.begin(),alphas.end());alphas.erase(unique(alphas.begin(),alphas.end(),[](double a,double b){return abs(a-b)<1e-9;}),alphas.end());for(double a:alphas){auto r=trials(n,a,50);auto ag=combine(r,50);double mean=0,se=0;if(ag.complete==100){vector<double>x(100);for(int j=0;j<100;j++)x[j]=accumulate(r[j].lengths.begin(),r[j].lengths.end(),0.)/50;mean=accumulate(x.begin(),x.end(),0.)/100;for(double v:x)se+=pow(v-mean,2);se=sqrt(se/9900.);}if(!first)cout<<',';first=false;cout<<"{\"n\":"<<n<<",\"alpha_times8\":"<<a<<",\"mean_first50\":";if(ag.complete==100)cout<<mean;else cout<<"null";cout<<",\"se_first50\":";if(ag.complete==100)cout<<se;else cout<<"null";cout<<",\"actual_steps\":"<<ag.steps<<",\"complete_runs\":"<<ag.complete<<",\"failures\":[";bool fc=false;for(int j=0;j<100;j++)if(r[j].failed>=0){if(fc)cout<<',';fc=true;cout<<"{\"run\":"<<j<<",\"episode\":"<<r[j].failed<<",\"actual_steps\":"<<r[j].steps<<"}";}cout<<"]}";cerr<<"early "<<n<<' '<<a<<" done "<<ag.complete<<'\n';}}
 cout<<"],\"fixture_n1_alpha_point5_first50\":";arr(fixture);
 // Own replacing-trace SARSA(lambda) illustration; original figure lambda unspecified.
 vector<double>w(DIM),trace(DIM);RNG start{1110001},act{1209174};long long steps=0;int completed=0;bool comma=false;cout<<",\"surfaces\":{\"lambda\":0.9,\"trace\":\"replacing\",\"alpha_times8\":0.3,\"seed\":1110001,\"frames\":[";
 auto frame=[&](const string&label){if(comma)cout<<',';comma=true;cout<<"{\"label\":\""<<label<<"\",\"steps\":"<<steps<<",\"completed_episodes\":"<<completed<<",\"weights\":";arr(w);vector<double>cost;for(int j=0;j<41;j++)for(int i=0;i<41;i++){State z{-1.2+1.7*i/40.,-.07+.14*j/40.};F f=base_features(z);double m=-1e300;for(int a=0;a<3;a++)m=max(m,value(w,action_features(f,a)));cost.push_back(i==40?0:-m);}cout<<",\"grid_cost\":";arr(cost);cout<<'}';};
 for(int e=0;e<9000;e++){fill(trace.begin(),trace.end(),0);State s{-.6+.2*start.next(),0};F f=base_features(s);int a=action(w,f,act);F fa=action_features(f,a);bool finished=false;for(int t=0;t<2000000;t++){State z=step(s,a);bool done=z.x>=.5;F g=base_features(z);int na=done?0:action(w,g,act);F ga=action_features(g,na);double delta=-1+(done?0:value(w,ga))-value(w,fa);for(int i:fa)trace[i]=1;for(int i=0;i<DIM;i++){w[i]+=.3/8*delta*trace[i];trace[i]*=.9;}steps++;if(steps==428)frame("Step428");if(done){finished=true;break;}s=z;a=na;fa=ga;}
  if(!finished){cerr<<"trace episode failed hard limit";return 4;}completed++;if(completed==12||completed==104||completed==1000||completed==9000)frame("Episode"+to_string(completed));}
 cout<<"],\"actual_steps\":"<<steps<<"}}\n";cerr<<"all finished\n";}
