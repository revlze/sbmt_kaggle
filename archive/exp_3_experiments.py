"""Breakthrough-stage diagnostics, kernel AUC ranker and learned RBF metric."""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

EPS = 1e-12

@dataclass
class KFold:
    repeat:int; fold:int; tr:np.ndarray; va:np.ndarray; ytr:np.ndarray; yva:np.ndarray
    Ktr:np.ndarray; Kva:np.ndarray

@dataclass
class KCache:
    folds:list; seeds:tuple; n:int

class KernelAUCRanker:
    def __init__(self, lambda_reg=1e-3, temperature=1., max_iter=150, hard_fraction=None):
        self.lambda_reg,self.temperature,self.max_iter,self.hard_fraction=lambda_reg,temperature,max_iter,hard_fraction
    def _opt(self,K,y,x0,pairs=None,max_iter=None):
        pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
        def fg(a):
            s=K@a; d=(s[pos,None]-s[neg][None,:])/self.temperature
            if pairs is None: mask=np.ones(d.shape,bool)
            else: mask=pairs
            dm=d[mask]; loss=np.logaddexp(0,-dm).mean()+self.lambda_reg*(a@(K@a))
            q=-expit(-dm)/(len(dm)*self.temperature); gs=np.zeros(len(y))
            ii,jj=np.nonzero(mask); np.add.at(gs,pos[ii],q); np.add.at(gs,neg[jj],-q)
            return loss, K.T@gs+2*self.lambda_reg*(K@a)
        return minimize(fg,x0,jac=True,method='L-BFGS-B',options={'maxiter':max_iter or self.max_iter,'ftol':1e-10})
    def fit(self,K,y):
        K=np.asarray(K,float); y=np.asarray(y); r=self._opt(K,y,np.zeros(len(y)))
        self.initial_loss_=float(np.log(2.0)); self.final_loss_=r.fun
        self.alpha_=r.x; self.K_fit_=K
        if self.hard_fraction:
            s=K@self.alpha_; p=np.flatnonzero(y==1); n=np.flatnonzero(y==0)
            d=s[p,None]-s[n][None,:]; cut=np.quantile(d,self.hard_fraction); mask=d<=cut
            r=self._opt(K,y,self.alpha_,mask,max(30,self.max_iter//2)); self.alpha_=r.x; self.final_loss_=r.fun
        return self
    def decision_function(self,K): return np.asarray(K)@self.alpha_
    @property
    def rkhs_norm_(self): return float(self.alpha_@(self.K_fit_@self.alpha_))

def check_kernel(K):
    assert np.isfinite(K).all() and np.allclose(K,K.T,atol=2e-5) and np.all(np.diag(K)>0)

def raw_kernel_cache(X,y,seeds,n_splits,C=1.,gamma=0.001,n_jobs=-1):
    X=np.asarray(X,float); y=np.asarray(y); jobs=[]
    for r,seed in enumerate(seeds):
        for f,(tr,va) in enumerate(StratifiedKFold(n_splits,shuffle=True,random_state=seed).split(X,y)): jobs.append((r,f,tr,va))
    def one(r,f,tr,va):
        sc=StandardScaler(); A=sc.fit_transform(np.log1p(X[tr])); B=sc.transform(np.log1p(X[va]))
        K=np.exp(-gamma*cdist(A,A,'sqeuclidean')); Kv=np.exp(-gamma*cdist(B,A,'sqeuclidean')); check_kernel(K)
        return KFold(r,f,tr,va,y[tr],y[va],K,Kv)
    return KCache(Parallel(n_jobs=n_jobs,prefer='threads')(delayed(one)(*j) for j in jobs),tuple(seeds),len(X))

def raw_svc_validation(X,y,Xv,yv,C,gamma):
    sc=StandardScaler(); A=sc.fit_transform(np.log1p(X)); B=sc.transform(np.log1p(Xv)); m=SVC(C=C,kernel='rbf',gamma=gamma).fit(A,y); s=m.decision_function(B); return roc_auc_score(yv,s),s

def current_kernel_cache(X,y,seeds,config,params,n_splits=5,n_jobs=1):
    from svc_experiments import LeakageSafeFeatureBuilder,_scale_blocks,FoldData,_kernel
    X=np.asarray(X,float); y=np.asarray(y); jobs=[]
    for r,seed in enumerate(seeds):
        for f,(tr,va) in enumerate(StratifiedKFold(n_splits,shuffle=True,random_state=seed).split(X,y)): jobs.append((r,seed,f,tr,va))
    def one(r,seed,f,tr,va):
        b=LeakageSafeFeatureBuilder(config,seed+f,n_splits); a,g,d,_,_=b.fit_transform(X[tr],y[tr]); av,gv,dv,_,_=b.transform(X[va])
        z=_scale_blocks(a,av,g,gv,d,dv); fd=FoldData(*z,y[tr],y[va]); K=_kernel(fd,params,True); Kv=_kernel(fd,params,False); check_kernel(K)
        return KFold(r,f,tr,va,y[tr],y[va],K,Kv)
    return KCache(Parallel(n_jobs=n_jobs,prefer='threads')(delayed(one)(*j) for j in jobs),tuple(seeds),len(X))

def svc_scores(cache,C,n_jobs=-1):
    def one(f):
        m=SVC(C=C,kernel='precomputed').fit(f.Ktr,f.ytr); s=m.decision_function(f.Kva); return f.repeat,f.va,roc_auc_score(f.yva,s),s
    return _collect(cache,Parallel(n_jobs=n_jobs,prefer='threads')(delayed(one)(f) for f in cache.folds))

def ranker_scores(cache,lambdas=(1e-4,1e-3,1e-2,.1),temperature=1.,hard_fraction=None,n_jobs=-1):
    best=None
    for lam in lambdas:
        def one(f):
            m=KernelAUCRanker(lam,temperature,120,hard_fraction).fit(f.Ktr,f.ytr); s=m.decision_function(f.Kva); return f.repeat,f.va,roc_auc_score(f.yva,s),s
        out=_collect(cache,Parallel(n_jobs=n_jobs,prefer='threads')(delayed(one)(f) for f in cache.folds))
        if best is None or out[0].mean()>best[0].mean(): best=(*out,{'lambda_reg':lam,'temperature':temperature,'hard_fraction':hard_fraction})
    return best

def _collect(cache,items):
    auc=[]; oof=np.full((len(cache.seeds),cache.n),np.nan)
    for r,idx,a,s in items: auc.append(a); oof[r,idx]=s
    return np.asarray(auc),oof

def fit_external_ranker(X,y,Xv,yv,kernel_kind,params,rank_params,config=None,seed=42):
    if kernel_kind=='raw':
        sc=StandardScaler(); A=sc.fit_transform(np.log1p(X)); B=sc.transform(np.log1p(Xv)); K=np.exp(-params['gamma']*cdist(A,A,'sqeuclidean')); Kv=np.exp(-params['gamma']*cdist(B,A,'sqeuclidean'))
    else:
        from svc_experiments import LeakageSafeFeatureBuilder,_scale_blocks,FoldData,_kernel
        b=LeakageSafeFeatureBuilder(config,seed,5); a,g,d,_,_=b.fit_transform(X,y); av,gv,dv,_,_=b.transform(Xv); z=_scale_blocks(a,av,g,gv,d,dv); fd=FoldData(*z,np.asarray(y),np.asarray(yv)); K=_kernel(fd,params,True); Kv=_kernel(fd,params,False)
    m=KernelAUCRanker(**rank_params).fit(K,y); s=m.decision_function(Kv); return roc_auc_score(yv,s),s

def duplicate_diagnostics(X,y):
    df=pd.DataFrame(np.asarray(X)); groups=df.groupby(list(df.columns),sort=False).indices; dup=[v for v in groups.values() if len(v)>1]
    return {'n_duplicate_groups':len(dup),'max_group_size':max(map(len,dup),default=1),'rows_in_duplicates':sum(map(len,dup)),'consistent_groups':sum(len(np.unique(np.asarray(y)[v]))==1 for v in dup)}

def nearest_diagnostics(X,y,X_cross=None):
    Z=StandardScaler().fit_transform(np.log1p(X)); nn=NearestNeighbors(n_neighbors=2).fit(Z); dist,idx=nn.kneighbors(Z); cos=NearestNeighbors(n_neighbors=2,metric='cosine').fit(Z).kneighbors(Z)[0][:,1]
    out=pd.DataFrame({'euclidean':dist[:,1],'cosine':cos,'same_target':np.asarray(y)==np.asarray(y)[idx[:,1]]})
    cross=None
    if X_cross is not None: cross=NearestNeighbors(n_neighbors=1).fit(Z).kneighbors(StandardScaler().fit(np.log1p(X)).transform(np.log1p(X_cross)))[0][:,0]
    return out,cross

def zero_pattern_diagnostics(X,y):
    M=np.packbits(np.asarray(X)==0,axis=1); _,inv,cnt=np.unique(M,axis=0,return_inverse=True,return_counts=True); rep=cnt[cnt>1]
    nn=NearestNeighbors(n_neighbors=2,metric='hamming').fit((np.asarray(X)==0)); h=nn.kneighbors((np.asarray(X)==0))[0][:,1]
    return {'repeated_masks':len(rep),'largest_group':int(rep.max()) if len(rep) else 1,'rows_repeated':int(rep.sum()),'median_nn_hamming':float(np.median(h))}

def triangle_search(X,powers=(.5,.75,1.,1.5,2.),n_triples=500000,coord_count=48,top_k=1000,seed=42,chunk=20000):
    X=np.asarray(X,float); rng=np.random.default_rng(seed); coords=rng.choice(X.shape[1],min(coord_count,X.shape[1]),False); rows=[]
    for p in powers:
        T=np.power(X,p); best=[]; remaining=n_triples
        for _ in range((remaining+chunk-1)//chunk):
            ids=np.array([rng.choice(len(X),3,False) for _ in range(min(chunk,remaining))]); remaining-=len(ids)
            A,B,C=T[ids[:,0]][:,coords],T[ids[:,1]][:,coords],T[ids[:,2]][:,coords]; R=np.abs(2*np.maximum.reduce([A,B,C])-A-B-C); s=np.median(R/(A+B+C+EPS),1)
            take=np.argpartition(s,min(top_k,len(s)-1))[:top_k]; best.extend(zip(s[take],ids[take].tolist()))
            if remaining<=0: break
        best=sorted(best)[:top_k]
        for _,ids in best:
            A,B,C=T[ids]; R=np.abs(2*np.maximum.reduce([A,B,C])-A-B-C); q=R/(A+B+C+EPS)
            rows.append({'row_i':ids[0],'row_j':ids[1],'row_k':ids[2],'power':p,'triangle_score':np.median(q),'q90':np.quantile(q,.9),'q99':np.quantile(q,.99),'fraction_good_coordinates':np.mean(q<1e-4)})
    return pd.DataFrame(rows).sort_values('triangle_score').reset_index(drop=True)

def triangle_null(X,power,n=20000,seed=42):
    rng=np.random.default_rng(seed); T=np.power(np.asarray(X,float),power); scores=[]
    for _ in range(n//1000):
        ids=np.array([rng.choice(len(T),3,False) for _ in range(1000)]); A,B,C=T[ids[:,0]],T[ids[:,1]],T[ids[:,2]]; R=np.abs(2*np.maximum.reduce([A,B,C])-A-B-C); scores.extend(np.median(R/(A+B+C+EPS),1))
    return np.quantile(scores,[.001,.01,.05])

def triangle_null_calibration(X,power,n=20000,seed=42):
    rng=np.random.default_rng(seed); X=np.asarray(X)
    perm_rows=np.vstack([row[rng.permutation(X.shape[1])] for row in X])
    shuffled=np.column_stack([rng.permutation(X[:,j]) for j in range(X.shape[1])])
    return {'random_triples':triangle_null(X,power,n,seed),'within_row_permutation':triangle_null(perm_rows,power,n,seed+1),'column_shuffle':triangle_null(shuffled,power,n,seed+2)}

def summary_row(name,family,aucs,oof,val,raw,current,current_oof,fit,params):
    d=np.asarray(aucs)-np.asarray(current); p=wilcoxon(d).pvalue if np.any(d) else 1.
    return {'experiment':name,'family':family,'CV_mean':np.mean(aucs),'CV_std':np.std(aucs),'CV_median':np.median(aucs),'delta_vs_raw':np.mean(np.asarray(aucs)-raw),'delta_vs_current':np.mean(d),'wins_vs_current':int((d>0).sum()),'losses_vs_current':int((d<0).sum()),'wilcoxon_vs_current':p,'validation_auc':val,'corr_with_current':spearmanr(np.asarray(oof).ravel(),np.asarray(current_oof).ravel()).statistic,'fit_time':fit,'params':params,'oof':oof,'auc_per_fold':list(aucs)}

class LearnedRBFMetric:
    def __init__(self,rank=0,lambda_identity=.1,lambda_lowrank=.1,epochs=80,lr=.03,sample_size=512,seed=42): self.rank,self.lambda_identity,self.lambda_lowrank,self.epochs,self.lr,self.sample_size,self.seed=rank,lambda_identity,lambda_lowrank,epochs,lr,sample_size,seed
    def fit(self,X,y):
        import torch
        torch.manual_seed(self.seed); X=np.asarray(X,np.float32); y=np.asarray(y); rng=np.random.default_rng(self.seed); idx=rng.choice(len(X),min(self.sample_size,len(X)),False)
        A=torch.tensor(X[idx]); ys=torch.tensor(2*y[idx]-1,dtype=torch.float32); self.delta_=torch.nn.Parameter(torch.zeros(X.shape[1])); self.U_=None if self.rank==0 else torch.nn.Parameter(torch.randn(X.shape[1],self.rank)*1e-3)
        opt=torch.optim.Adam([self.delta_]+([] if self.U_ is None else [self.U_]),lr=self.lr)
        Y=ys[:,None]*ys[None,:]; Y=Y-Y.mean(0)-Y.mean(1)[:,None]+Y.mean()
        for _ in range(self.epochs):
            w=torch.exp(self.delta_); V=A*torch.sqrt(w); D=torch.cdist(V,V).square()
            if self.U_ is not None: D=D+torch.cdist(A@self.U_,A@self.U_).square()
            K=torch.exp(-D/X.shape[1]); Kc=K-K.mean(0)-K.mean(1)[:,None]+K.mean(); align=(Kc*Y).sum()/(Kc.norm()*Y.norm()+1e-9)
            loss=-align+self.lambda_identity*self.delta_.square().mean()+(0 if self.U_ is None else self.lambda_lowrank*self.U_.square().mean())
            opt.zero_grad(); loss.backward(); opt.step()
        self.weights_=np.exp(self.delta_.detach().numpy()); self.U_array_=None if self.U_ is None else self.U_.detach().numpy(); self.gradient_norm_=float(self.delta_.grad.norm()); return self
    def kernel(self,A,B):
        A=np.asarray(A); B=np.asarray(B); V=A*np.sqrt(self.weights_); W=B*np.sqrt(self.weights_); D=cdist(V,W,'sqeuclidean')/A.shape[1]
        if self.U_array_ is not None: D+=cdist(A@self.U_array_,B@self.U_array_,'sqeuclidean')/A.shape[1]
        return np.exp(-D)

def learned_metric_scores(X,y,seeds,rank,lambda_identity,lambda_lowrank=.1,n_splits=5,n_jobs=-1,seed=42):
    X=np.asarray(X,float); y=np.asarray(y); jobs=[]
    for r,s in enumerate(seeds):
        for f,(tr,va) in enumerate(StratifiedKFold(n_splits,shuffle=True,random_state=s).split(X,y)): jobs.append((r,f,tr,va,s))
    def one(r,f,tr,va,s):
        sc=StandardScaler(); A=sc.fit_transform(np.log1p(X[tr])); B=sc.transform(np.log1p(X[va])); m=LearnedRBFMetric(rank,lambda_identity,lambda_lowrank,seed=s+f).fit(A,y[tr]); K=m.kernel(A,A); Kv=m.kernel(B,A); check_kernel(K); clf=SVC(C=10,kernel='precomputed').fit(K,y[tr]); z=clf.decision_function(Kv); return r,va,roc_auc_score(y[va],z),z,m
    items=Parallel(n_jobs=n_jobs,prefer='threads')(delayed(one)(*j) for j in jobs); auc,oof=_collect(KCache([],tuple(seeds),len(X)),[(a,b,c,d) for a,b,c,d,_ in items]); return auc,oof,[x[-1] for x in items]

def learned_metric_validation(X,y,Xv,yv,rank,lambda_identity,lambda_lowrank=.1,seed=42):
    sc=StandardScaler(); A=sc.fit_transform(np.log1p(X)); B=sc.transform(np.log1p(Xv)); m=LearnedRBFMetric(rank,lambda_identity,lambda_lowrank,seed=seed).fit(A,y); K=m.kernel(A,A); Kv=m.kernel(B,A); clf=SVC(C=10,kernel='precomputed').fit(K,y); s=clf.decision_function(Kv); return roc_auc_score(yv,s),s,m
