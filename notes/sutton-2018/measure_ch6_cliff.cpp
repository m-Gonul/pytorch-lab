// Original independent reproduction of Sutton Figure6.3, all full source budgets.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
struct Random { uint64_t s; explicit Random(uint64_t seed):s(seed?seed:1){} uint64_t next(){s^=s>>12;s^=s<<25;s^=s>>27;return s*2685821657736338717ULL;} double u(){return (next()>>11)*0x1.0p-53;} int integer(int n){return int(u()*n);} };
using QTable=std::array<std::array<double,4>,48>;
int action(const QTable&q,int s,Random&r){if(r.u()<.1)return r.integer(4);double m=q[s][0];int ties[4]={0},n=1;for(int a=1;a<4;a++){if(q[s][a]>m){m=q[s][a];ties[0]=a;n=1;}else if(q[s][a]==m)ties[n++]=a;}return ties[r.integer(n)];}
std::pair<int,double> step(int s,int a){static int dr[4]={-1,0,1,0},dc[4]={0,1,0,-1};int row=std::clamp(s/12+dr[a],0,3),col=std::clamp(s%12+dc[a],0,11);if(row==3&&col>=1&&col<=10)return {36,-100.};return {row*12+col,-1.};}
struct Stats {double mean=0,se=0;uint64_t steps=0;long maxEpisode=0;};
Stats experiment(int method,double alpha,int runs,int episodes,uint64_t seed){Stats out;double ss=0;for(int run=0;run<runs;run++){Random r(seed+104729ULL*(run+1)+1000003ULL*method+1000033ULL*int(std::round(alpha*1000)));QTable q{};double total=0;for(int ep=0;ep<episodes;ep++){int s=36,a=action(q,s,r);long steps=0;for(;;){auto[n,reward]=step(s,a);total+=reward;out.steps++;steps++;if(steps>10000000)throw std::runtime_error("Episode did not terminate within explicit safety budget");int nextAction=-1;double target=reward;if(n!=47){if(method==0){nextAction=action(q,n,r);target+=q[n][nextAction];}else if(method==1)target+=*std::max_element(q[n].begin(),q[n].end());else {double sum=0;for(auto x:q[n])sum+=x;target+=.9* *std::max_element(q[n].begin(),q[n].end())+.1*sum/4;}}q[s][a]+=alpha*(target-q[s][a]);if(n==47)break;s=n;a=method==0?nextAction:action(q,s,r);}out.maxEpisode=std::max(out.maxEpisode,steps);}double y=total/episodes;out.mean+=y/runs;ss+=y*y/runs;}out.se=runs>1?std::sqrt(std::max(0.,(ss-out.mean*out.mean)/(runs-1))):0;return out;}
int main(int argc,char**argv){if(argc<2){std::cerr<<"Usage: measure_ch6_cliff OUTPUT.json [interim_runs=50000] [asymptotic_runs=10]\n";return 1;}int interim=argc>2?std::atoi(argv[2]):50000,asymptotic=argc>3?std::atoi(argv[3]):10;std::ofstream f(argv[1]);f<<std::setprecision(17)<<"{\n\"seed\":1,\"epsilon\":0.1,\"gamma\":1,\"initial_Q\":0,\"tie\":\"uniform greedy ties\",\"source\":\"Figure6.3: interim 50000 runs x100 episodes; asymptotic10 runs x100000 episodes. Alpha grid0.1..1 step0.05 is our explicit sampling of the source abscissa.\",\n\"rows\":[\n";bool first=true;for(int method=0;method<3;method++)for(int k=2;k<=20;k++)for(int mode=0;mode<2;mode++){int runs=mode?asymptotic:interim,episodes=mode?100000:100;auto o=experiment(method,k/20.,runs,episodes,1);if(!first)f<<",\n";first=false;f<<"{\"method\":"<<method<<",\"alpha\":"<<k/20.<<",\"runs\":"<<runs<<",\"episodes\":"<<episodes<<",\"mean\":"<<o.mean<<",\"se_across_runs\":"<<o.se<<",\"steps\":"<<o.steps<<",\"max_episode\":"<<o.maxEpisode<<"}";f.flush();std::cerr<<"method="<<method<<" alpha="<<k/20.<<" episodes="<<episodes<<" runs="<<runs<<" reward="<<o.mean<<" steps="<<o.steps<<"\n";}f<<"\n]}\n";}
