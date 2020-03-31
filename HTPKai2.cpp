#include "StopWatch.h"
#include "HTPKai2.h"

GivensRotationSequence allocateGivensRotationSequence(int m, int n, vi cols){
    //* m - n + 2*(n - 1 - j) Givens Rotations for each j:cols
    int length = 0, t0 = m - n, t1 = n - 1;
    for(int j: cols){
        length += t0 + 2 * (t1 - j);
    }
    return GivensRotationSequence(length);
}

void getAndApplyGivensRotation(MatrixXd &A, vi cols, int j, double p, double q, int i1, int i2){
    //* Calculate and apply givens
    //! 0-based
    
}

void QrRecoverOneCol(MatrixXd &A, vi &cols, int j, GivensRotationSequence &Gseq){
    //! The lower triangular part of A are assumed to be zero except in column j
    //! j is not in cols
    //! 0-based
    //* Preparations
    int n = A.cols(), m = A.rows();
    
    //* Stage 1: process rows "i > n" (last column will not be effected)
    for(int i = m-1; i > n; i--){
        JacobiRotation<double> G;
        G.makeGivens(A(i-1, j), A(i, j));
        A.col(j).applyOnTheLeft(i-1, i, G.transpose());

        for(int c: cols){
            A.col(c).applyOnTheLeft(i-1, i, G.transpose());
        }
        Gseq.push(GivensRotation(G, i-1, i));
    }
    //cout << "========B1test after Stage1" <<endl; cout << A << endl;
    
    //* Things are almost done if the changed column is the last one
    if(j == n-1){
        JacobiRotation<double> G;
        G.makeGivens(A(n-1, j), A(n, j));
        A.col(j).applyOnTheLeft(n-1, n, G.transpose());

        for(int c: cols){
            A.col(c).applyOnTheLeft(n-1, n, G.transpose());
        }
        Gseq.push(GivensRotation(G, n-1, n));
        return ;
    }
    //* Stage 2: continue on process rows "j+2 <= i <= n"
    for(int i = n, r=1; i > j+1; i--, r++){
        JacobiRotation<double> G;
        G.makeGivens(A(i-1, j), A(i, j));
        A.col(j).applyOnTheLeft(i-1, i, G.transpose());
        A.rightCols(r).applyOnTheLeft(i-1, i, G.transpose());

        for(int c: cols){
            A.col(c).applyOnTheLeft(i-1, i, G.transpose());
        }
        Gseq.push(GivensRotation(G, i-1, i));
        // cout << "========B1test inside Stage2" <<endl; cout << A << endl;
    }

    // cout << "========B1test after Stage 2" <<endl; cout << A << endl;
    //* Stage 3: process sub-diagonal fill-ins
    for(int jr = j; jr < n; jr++){
        JacobiRotation<double> G;
        G.makeGivens(A(jr, jr), A(jr+1, jr));
        A.rightCols(n - jr).applyOnTheLeft(jr, jr+1, G.transpose());

        for(int c: cols){
            A.col(c).applyOnTheLeft(jr, jr + 1, G.transpose());
        }
        Gseq.push(GivensRotation(G, jr, jr + 1));
        //cout << "========B1test inside Stage3" <<endl; cout << A << endl;
    }
    return;
}

void QrRecover(MatrixXd &A, vi &cols, GivensRotationSequence &Gseq){
    //* "A" is an upper-triangular matrix with columns in "cols" replaced (with arbitrary vectors)
    //* This function recovery A to be upper-triangular again using a sequence of Givens Rotations,
    //* which are stored in "Gseq".
    //! The lower triangular part of A are assumed to be zero except in column j: cols
    //! 0-based

    //* Preparations
    int n = A.cols(), m = A.rows();
    vi tmp_cols(cols);

    sort(ALL(tmp_cols));  //* for security we sort tmp_cols
    //* The Qr recovery process start from the right-most column,
    //* then go left one column by one column until done.
    while(!tmp_cols.empty()){
        int j = tmp_cols.back();
        tmp_cols.pop_back();
        QrRecoverOneCol(A, tmp_cols, j, Gseq);
    }

    return;
}

MatrixXd LsReference(MatrixXd const &A, MatrixXd const &b){
    HouseholderQR<MatrixXd> qr(A);
    return qr.solve(b);
}

MatrixXd LsRecover(MatrixXd const &A, MatrixXd const &b, vi const &cols, HouseholderQR<MatrixXd> &qr_base){
    //* preparations
    int m = A.rows();
    int n = gK; //n = A.cols();
    MatrixXd R_recover(m, n), b_tmp(b);
    //* replace columns in base R, and upper-triangular is spoiled
    R_recover = qr_base.matrixQR().triangularView<Upper>();
    //! COST ABOUT 80% OF RUNTIME IN THE FOLLOWING LOOP
    // MatrixXd v;
    REP(i, cols.size()){
        int j = cols[i];
        //* OLD IMPLEMENTATION
        // v = A.col(j);
        // v.applyOnTheLeft(qr_base.householderQ().transpose());
        // R_recover.col(j) = v;
        
        //* New test
        //(A.col(j)).applyOnTheLeft(qr_base.householderQ().transpose());
        //! NEED MODIFICATION
        R_recover.col(j) = A.col(i);
        //? R_recover.col(j).applyOnTheLeft(qr_base.householderQ().transpose());
    }
    //* utilize QrRecover to recover the spoiled upper-triangular matrix
    GivensRotationSequence Gseq = allocateGivensRotationSequence(m, n, cols);
    vi cols_tmp(cols);
    QrRecover(R_recover, cols_tmp, Gseq);
    //* process right-hand-side b_tmp (copy of b)
    b_tmp.applyOnTheLeft(qr_base.householderQ().transpose());
    
    for(int i=0; i< Gseq.end_pt; i++){
       b_tmp.applyOnTheLeft(Gseq.data[i].i1, Gseq.data[i].i2, Gseq.data[i].G.transpose());
    }
    
    //* solving upper-triangular system
    MatrixXd x;
    x = R_recover.block(0, 0, n, n).triangularView<Upper>().solve(b_tmp.topRows(n));
    return x;
}

MatrixXd LsSequenceInner(HouseholderQR<MatrixXd> &qr, MatrixXd const &b){
    //* initialization
    int m = qr.rows(), n = qr.cols();
    MatrixXd X = ArrayXXd::Zero(n, n).matrix();
    MatrixXd b_tmp(b); 

    b_tmp.applyOnTheLeft(qr.householderQ().transpose()); //* Q^T * b can be reused
    //* solve sequential least square problems
    REP(j, n){
        X.block(0, j, j+1, 1)
            = qr.matrixQR().block(0, 0, j+1, j+1).triangularView<Upper>().solve(b_tmp.topRows(j+1));
    }
    return X;
}

MatrixXd LsSequence(MatrixXd const &A, MatrixXd const &b){
    //* perform qr decomposition on A
    HouseholderQR<MatrixXd> qr(A); //! NEED MODIFICATION
    return LsSequenceInner(qr, b);
    /* *
    //* preparations
    int m = A.rows(), n = A.cols();
    MatrixXd X = ArrayXXd::Zero(n, n).matrix();
    MatrixXd b_tmp(b); 
    
    b_tmp.applyOnTheLeft(qr.householderQ().transpose()); //* Q^T * b can be reused
    //* solve sequential least square problems
    REP(j, n){
        X.block(0, j, j+1, 1)
            = qr.matrixQR().block(0, 0, j+1, j+1).triangularView<Upper>().solve(b_tmp.topRows(j+1));
    }
    return X;
    */
}

MatrixXd LsSequenceRef(MatrixXd const &A, MatrixXd const &b){
    //* preparations
    int m = A.rows(), n = A.cols();
    MatrixXd X = ArrayXXd::Zero(n, n).matrix();
    //* solve sequential least square problems
    REP(j, n){
        MatrixXd Atmp = A.leftCols(j+1);
        X.block(0, j, j+1, 1)
            = Atmp.householderQr().solve(b);
    }
    return X;
}

mat projectToFull(const mat &x_s, const vi &seq){
    mat x = MatrixXd::Zero(gN, 1);
    REP(i, seq.size()){
        x(seq[i], 0) = x_s(i, 0);
    }
    return x;
}

//* input: vector (in Eigen mat form) v
//* output: support S, sequence of index seq
void calHardThreshold(const mat&v, Supp &S, vi &seq){
    //cout<<"v: " << v.array().abs().transpose()<<endl;
    vector<pdi> v_abs(gN);// first: abs(v[idx]), second: idx
    REP(i, gN) {
        v_abs[i].first = abs(v(i, 0)); 
        v_abs[i].second = i;
    }
    sort(ALL(v_abs), greater<pdi>());
    /**
    REP(i, gN){
        seq[i] = v_abs[i].second;
        if(i < gK){
            S = S | (gOne << v_abs[i].second);
        }
    }
    */
    S.reset();
    REP(i, gK){
        seq[i] = v_abs[i].second;
        S = S | (gOne << v_abs[i].second);
    }
}
/**
 * calGradStep
 * * calculate gradient step
 * @param x of size gN-by-1, support sequence seq, input
 * @param seq support sequence, input
 * @param r of size gM-by-1, residual, output
 * @param v of size gN-by-1, gradient step result, output
 * * r = gRhs - gA * x, v = x + gA^T * r
 */
void calGradStep(mat &x, vi &seq, mat &r, mat &v){
    r = gRhs;
    for(int idx: seq) 
        r = r - gA.col(idx) * x(idx, 0);
    v = x + gA.transpose() * r; //* v = x + gA^T * r
}

int HTPStepScratch(Supp &S, vi &seq, Supp &S_out, vi &seq_out){
    QrData qr_data_tmp; gQrInfo.push_back(qr_data_tmp);
    QrData &qr_data = gQrInfo.back();
    //* extract the corresponding of sensing matrix
    mat A(gM, gK);
    REP(i, gK) A.col(i) = gA.col(seq[i]); 
    //* QR decomposition 
    HouseholderQR<MatrixXd> &qr = qr_data.qr;
    qr.compute(A);
    //* solve full least square
    mat x_s = qr.solve(gRhs); 
    //cout<<"x_s: "<<x_s.transpose()<<endl;
    //* project x_s to full x
    mat x = projectToFull(x_s, seq);
    mat r(gM, 1), v(gN, 1);
    calGradStep(x, seq, r, v);
    //cout<<"r: "<<r.transpose()<<endl;
    //* prepare for output using hard threshold
    calHardThreshold(v, S_out, seq_out);
    //* update global database
    qr_data.S = S;
    qr_data.QTransA = gA;
    qr_data.QTransA.applyOnTheLeft(qr.householderQ().transpose());
    qr_data.Seq = seq;
    // qr_data.ref_seq = vi(gN);
    // REP(i, gN) qr_data.ref_seq[i] = v_abs[i].second;
    //gVisitedSupp.insert(S_out); //WRONG

    int flag = 1;
    if(r.norm() <= gResTol) flag = 0; //* convergence confirmed
    else{ //* visited S
        gVisitedSupp.insert(S);
    }
    return flag;
}

ll calRefScore(Supp &S, vi &seq_ref){
    ll score = 0;
    REP(i, gK){
        int j = seq_ref[i];
        if(((S >> j) & gOne) == 0) 
            score += (gK - i) * (gK - i);
    }
    return score;
}

void printSupp(const Supp &S){ 
    string str = S.to_string();
    str = str.substr(str.length()-gN);
    reverse(ALL(str));
    cout<<str<<endl; 
}

template <class T>
void printVec(const vector<T> &a){
    REP(i, a.size())  cout << a[i] << " ";
    cout << endl;
}

//* calculate Humming distance
int calHammingDist(Supp &S, Supp &S_ref){
    Supp tmp = S^S_ref;
    return tmp.count()/2;
}

//* calculate smallest Humming distance to fixed points
int calHamDistClosest(Supp &S){
    int res = gN;
    for(auto supp_fp: gFixedPoint){
        res = min(res, calHammingDist(S, supp_fp));
    }
    return res;
}

template <class T>
vi sortIdxDescent(const vector<T> &a){
    int n = a.size();
    vector<pair<T, int> > r(n);
    REP(i, n){
        r[i].first = a[i];
        r[i].second = i;
    }
    sort(ALL(r)); reverse(ALL(r));
    vi res(n);
    REP(i, n) res[i] = r[i].second;
    return res;
}

//* check whether support S is worth using LsRecover subroutine to solve 
bool isSuppNearQrInfoData(Supp &S){
    int maxDist = gK * gSuppDiffRatio;
    for(auto qr_data: gQrInfo){
        int dist = calHammingDist(S, qr_data.S);
        if(dist <= maxDist) return true;
    }
    return false;
}

int HTPStepReference(Supp &S, vi &seq, Supp &S_out, vi &seq_out){
    if(gQrInfo.size()==0) return -1;
    //* decide best reference QR Data
    vector<pair<ll, int>> score_array;
    for(int idx=0; idx < gQrInfo.size(); idx++){
        QrData &qr_data = gQrInfo[idx];
        ll score = calRefScore(S, qr_data.Seq);
        score_array.push_back(make_pair(score, idx));
    }
    sort(ALL(score_array));
    QrData &qr_data_ref = gQrInfo[score_array[0].second];
    //* indexSwapIn = \bar{Sref} intersect S
    //* indexSwapOut = \bar{S} intersect Sref
    vi indexSwapIn, indexSwapOut;
    vi &seq_ref = qr_data_ref.Seq;
    Supp &Sref = qr_data_ref.S;
    //* decide index swapin/swapout
    vi cols;
    for(int i=0; i < gK; i++){
        int idx = seq_ref[i];
        //! CHECK right shift valid
        if(((S >> idx) & gOne) == 0){//* idx not in S means it should be swap out
            indexSwapOut.push_back(idx);
            cols.push_back(i);
        }
    }
    for(auto idx: seq){
        //! CHECK right shift valid
        if(((Sref >> idx) & gOne) == 0){//* idx not in Sref means it should be swapped in
            indexSwapIn.push_back(idx);
        }
    }
    int num_change = indexSwapOut.size();
    mat A_change(gM, num_change);
    REP(i, num_change){
        A_change.col(i) = qr_data_ref.QTransA.col(indexSwapIn[i]);
    }
    mat x_s = LsRecover(A_change, gRhs, cols, qr_data_ref.qr);
    //cout<<"x_s: "<<x_s.transpose()<<endl;

    vi seq_new(ALL(seq_ref));
    REP(i, num_change){
        seq_new[cols[i]] = indexSwapIn[i];
    }

    mat x = projectToFull(x_s, seq_new);
    mat r(gM, 1), v(gN, 1);
    calGradStep(x, seq_new, r, v);
    //cout<<"r: "<<r.transpose()<<endl;
    
    //* prepare for output using hard thresholding
    calHardThreshold(v, S_out, seq_out);
    //* update global database
    //gVisitedSupp.insert(S_out); //WRONG

    int flag = 1;
    if(r.norm() <= gResTol) flag = 0; //* convergence confirmed
    else{ //* visited S
        gVisitedSupp.insert(S);
    }
    return flag;
}

int HTPStep(Supp &S, vi &seq, Supp &S_out, vi &seq_out){
    int flag = 0;
    return flag;
}

//! IMPORTANT: We suppose input supp length nK
//! return value flag = 0: convergence confirmed
//! return value flag = 1: continue searching
//! return value flag =-1: abandon this branch
int HTPInner(Supp &S, vi &seq, Supp &S_out, vi &seq_out){
    int flag = 1; 
    // cout << gQrInfo.size()<<" ";
    //* 0. check input validity
    if(gVisitedSupp.count(S)) return -1; //* already visted

    gHTPCnt ++;
    //* 1. 
    Supp S_old, S_new;
    vi seq_old(seq), seq_new(seq);
    S_old = S;
    while(1){
        bool useRef = isSuppNearQrInfoData(S_old);
        //* trigger HTP_STEP with needed version
        int step_flag;
        if(useRef){
            step_flag = HTPStepReference(S_old, seq_old, S_new, seq_new);

        } else{ //* compute QR decompositio from scratch
            step_flag = HTPStepScratch(S_old, seq_old, S_new, seq_new);
        }
        //* process according to return value
        if(!step_flag){ //* step_flag == 0, convergence confirmed
            flag = 0; break;
        }
        else if(S_old == S_new){ //* fixed point reached
            // gFixedPoint.insert(S_old);
            flag = 2; break;
        }
        else if(gVisitedSupp.count(S_new)){//* visited support reached, end this brunch
            flag = -1; break;
        }
        //* continue to the next HTP iteration
        //printSupp(S_new);
        gVisitedSupp.insert(S_old); //! DELETE?
        S_old = S_new;
        REP(i, gK) seq_old[i] = seq_new[i];
    }

    //* 9. process output
    //printSupp(S_new);
    S_out = S_new; //* technically S_out is not visited
    if(seq_out.size() < gK) seq_out.resize(gK);
    REP(i, seq_out.size()) seq_out[i] = seq_new[i];
    return flag;
}

//* input: S, seq --> fixed point
//* output: arrS, arrSeq
int makeSeed(Supp &S, vi &seq, vector<Supp> &arrS, vector<vi> &arrSeq){
    // cout<<"input seq: "; printVec(seq);
    //* 1. query gQrInfo for existence of QR decomposition
    bool existed = false;
    int qIdx;
    for(qIdx=0; qIdx < gQrInfo.size(); qIdx++){
        if(gQrInfo[qIdx].S == S){
            existed = true;
            break;
        }
    }
    //* 2. calculate sequencial least square solutions
    mat X;
    if(existed){ //* directly use the exited qr_data
        HouseholderQR<MatrixXd> &qr = gQrInfo[qIdx].qr;
        seq = gQrInfo[qIdx].Seq;
        // cout<<"modif seq: "; printVec(seq);
        X = LsSequenceInner(qr, gRhs); //* X: gN-by-gK
        // cout<<"X: "<< X <<endl;
    }else{ //* compute full QR decomposition and add to gQrInfo
        QrData new_data;
        gQrInfo.push_back(new_data);
        QrData &qr_data = gQrInfo.back();
        qr_data.S = S;
        qr_data.Seq = seq;
        mat A_tmp(gM, gK);
        REP(i, gK) A_tmp.col(i) = gA.col(seq[i]);
        qr_data.qr.compute(A_tmp);
        qr_data.QTransA = gA;
        qr_data.QTransA.applyOnTheLeft(qr_data.qr.householderQ().transpose());
        X = LsSequenceInner(qr_data.qr, gRhs); //* X: gN-by-gK
    }
    //* 3. Perform gradient step and hard thresholding for each solution x, and 
    //* calculate candidate supports
    int iLow = gSuppGuardRatio*gK;
    vi seq_tmp;
    Supp S_tmp=gZeroS;

    mat x_tmp, r_tmp(gM, 1), v_tmp(gN, 1);
    for(int i = 0; i< gK; i++){
        seq_tmp.push_back(seq[i]);
        S_tmp = S_tmp | (gOne << seq[i]); 
        if(i < iLow) continue;
        x_tmp = projectToFull(X.col(i), seq_tmp);
        calGradStep(x_tmp, seq_tmp, r_tmp, v_tmp);
        //* perform hard thresholding
        Supp S_out = gZeroS;
        vi seq_out(gK, 0);
        calHardThreshold(v_tmp, S_out, seq_out);
        //cout<<x_tmp.transpose()<<endl;
        //printVec(seq_out);
        //* add result to output
        if(!gVisitedSupp.count(S_out)){
            arrS.push_back(S_out);
            arrSeq.push_back(seq_out);
            //gVisitedSupp.insert(S_out); // WRONG
        }
    }
    arrS.pop_back(); arrSeq.pop_back();
    int flag = 0;
    return flag;
}

void setGlobalVar(){
}

int HTPOriginal(Supp &S_new, vi & seq_new){
    int htp_flag;
    //* first step
    mat v = gA.transpose() * gRhs; 
    Supp S_old;
    vi seq_old(gK, 0);
    calHardThreshold(v, S_old, seq_old); //! NO least squares done
    // cout<<"S_old: "; printSupp(S_old);
    htp_flag = HTPInner(S_old, seq_old, S_new, seq_new);
    if(htp_flag == 2) gFixedPoint.insert(S_new);
    return htp_flag;
}

vi sortSuppCandid(vector<Supp> &candidS, vector<vi> &candidSeq){
    int num = candidS.size();
    vector<pi> record(num);
    REP(i, num){
        record[i].first = calHamDistClosest(candidS[i]);
        record[i].second = -i;
    }
    /*
    REP(i, num) cout<< record[i].first<<" ";
    cout<<endl;
    */
    sort(ALL(record)); reverse(ALL(record));
    vi res(num);
    REP(i, num) res[i] = -record[i].second;
    return res;
}


int dfsHTPKai(Supp &S_in, vi &seq_in){
    int dfs_flag = -1;
    Supp S_old, S_new;
    vector<Supp> arrS, candidS;
    vi seq_old, seq_new;
    vector<vi> arrSeq, candidSeq;
    int seed_flag = makeSeed(S_in, seq_in, arrS,  arrSeq);
    int htp_flag;
    int idx = 0;
    int iDrop = gSuppDropRatio * gK;
    while(!arrS.empty()){
        S_old = arrS.back();
        seq_old = arrSeq.back();
        arrS.pop_back(); arrSeq.pop_back();
        //cout<<"Input seq: "; printVec(seq_old);
        
        htp_flag = HTPInner(S_old, seq_old, S_new, seq_new);
        
        if(htp_flag == -1) continue;
        if(htp_flag == 0){
            gResultS = S_new;
            return 0;
        }
        if(idx < iDrop) {idx ++ ; continue;}
        if(htp_flag == 2){
            gFixedPoint.insert(S_new);
            
            printSupp(S_new);
            
            int tmp_dfs_flag = dfsHTPKai(S_new, seq_new);
            return tmp_dfs_flag;
        }
        idx++;
        
    }
    return dfs_flag;
}

int dfsHTPKaiMaxDistLocal(Supp &S_in, vi &seq_in){
    int dfs_flag = -1;
    Supp S_old, S_new;
    vector<Supp> arrS, candidS;
    vi seq_old, seq_new, candidDiffDist;
    vector<vi> arrSeq, candidSeq;
    int seed_flag = makeSeed(S_in, seq_in, arrS,  arrSeq);
    int htp_flag;

    while(!arrS.empty()){
        S_old = arrS.back();
        seq_old = arrSeq.back();
        arrS.pop_back(); arrSeq.pop_back();
        if(!arrS.empty()) {arrS.pop_back(); arrSeq.pop_back();}
        //if(!arrS.empty()) {arrS.pop_back(); arrSeq.pop_back();}
        //cout<<"Input seq: "; printVec(seq_old);
        htp_flag = HTPInner(S_old, seq_old, S_new, seq_new);
        if(htp_flag == 0){
            gResultS = S_new;
            return 0;
        }
        if(gHTPCnt >= gMaxHTPCnt) return 99;
        if(htp_flag == 2){
            candidS.push_back(S_new);
            candidSeq.push_back(seq_new);
            candidDiffDist.push_back(min(calHammingDist(S_old, S_new), calHamDistClosest(S_new)));
        }
    }
    if(candidS.empty()) return -1;
    vi sortIdx = sortIdxDescent(candidDiffDist);
    for(auto idx: sortIdx) cout<<idx<<" "; cout<<endl;
    //cout<<"candidate len: "<<candidS.size()<<endl;
    //cout<<"max idx: "<<sortIdx[0]<<endl;
    REP(i, candidS.size()) gFixedPoint.insert(candidS[i]);
    for(auto i:sortIdx){
        //printSupp(candidS[i]);
        int tmp_dfs_flag = dfsHTPKaiMaxDistLocal(candidS[i], candidSeq[i]);
        
        if(tmp_dfs_flag == 0) return 0;
        if(tmp_dfs_flag ==99) return 99;
        if(tmp_dfs_flag < 0) continue;
    }
    if(gHTPCnt >= gMaxHTPCnt) return 99;
    return dfs_flag;
}

int dfsHTPKaiMaxDist(Supp &S_in, vi &seq_in){
    int dfs_flag = -1;
    Supp S_old, S_new;
    vector<Supp> arrS, candidS;
    vi seq_old, seq_new;
    vector<vi> arrSeq, candidSeq;
    int seed_flag = makeSeed(S_in, seq_in, arrS,  arrSeq);
    int htp_flag;

    while(!arrS.empty()){
        S_old = arrS.back();
        seq_old = arrSeq.back();
        arrS.pop_back(); arrSeq.pop_back();
        if(!arrS.empty()) {arrS.pop_back(); arrSeq.pop_back();}
        //if(!arrS.empty()) {arrS.pop_back(); arrSeq.pop_back();}
        //cout<<"Input seq: "; printVec(seq_old);
        htp_flag = HTPInner(S_old, seq_old, S_new, seq_new);
        if(htp_flag == 0){
            gResultS = S_new;
            return 0;
        }
        if(gHTPCnt >= gMaxHTPCnt) return 99;
        if(htp_flag == 2){
            candidS.push_back(S_new);
            candidSeq.push_back(seq_new);
        }
    }
    if(candidS.empty()) return -1;

    vi sortIdx = sortSuppCandid(candidS, candidSeq);
    
    //cout<<"candidate len: "<<candidS.size()<<endl;
    //cout<<"max idx: "<<sortIdx[0]<<endl;
    REP(i, candidS.size()) gFixedPoint.insert(candidS[i]);
    for(auto i:sortIdx){
        //printSupp(candidS[i]);
        int tmp_dfs_flag = dfsHTPKaiMaxDist(candidS[i], candidSeq[i]);
        
        if(tmp_dfs_flag == 0) return 0;
        if(tmp_dfs_flag ==99) return 99;
        if(tmp_dfs_flag < 0) continue;
    }
    if(gHTPCnt >= gMaxHTPCnt) return 99;
    return dfs_flag;
}

//! gM, gN, gK should be assigned before call this function
//! will update global variables
void genGaussMatAndShortSignal(){
    //* random generator settings
    random_device rd;
    mt19937 gen(rd());  
    normal_distribution<double> dis(0, 1.0);
    gA = Eigen::MatrixXd::Zero(gM,gN).unaryExpr([&](double dummy){return dis(gen);});
    REP(j, gN) gA.col(j).normalize();
    mat x_sm = Eigen::MatrixXd::Zero(gK, 1).unaryExpr([&](double dummy){return dis(gen);});
    mat x_orig = Eigen::MatrixXd::Zero(gN, 1);
    REP(i, gK) x_orig(i, 0) = x_sm(i, 0)>0?(x_sm(i, 0)+0.01): (x_sm(i, 0)-0.01);
    gRhs = gA * x_orig;
    //* update global parameters
    gRhsNorm = gRhs.norm();
    gResTol = gRhsNorm * gResTolRatio;
    return;
}

void clearGlobalRecord(){
    gHTPCnt = 0;
    gVisitedSupp.clear();
    gFixedPoint.clear();
    gQrInfo.clear();
    return;
}

void testHTPKaiRand(){
    cout<< "Please input compressive sensing problem sizes: (m, n, k)" <<flush;
    int m, n, k; cin >> m >> n >> k;
    //m = 300, n = 400, k = 160;
    //* random generator settings
    random_device rd;
    mt19937 gen(rd());  
    normal_distribution<double> dis(0, 1.0);
    //* declare random matrix
    gA = Eigen::MatrixXd::Zero(m,n).unaryExpr([&](double dummy){return dis(gen);});
    REP(j, n) gA.col(j).normalize();

    mat x_sm = Eigen::MatrixXd::Zero(k, 1).unaryExpr([&](double dummy){return dis(gen);});
    mat x_orig = Eigen::MatrixXd::Zero(n, 1);
    REP(i, k) x_orig(i, 0) = x_sm(i, 0)>0?(x_sm(i, 0)+0.01): (x_sm(i, 0)-0.01);
    gRhs = gA * x_orig;

    gM = m, gN = n, gK = k;

    gRhsNorm = gRhs.norm();
    gResTol = gRhsNorm * gResTolRatio;
    //gSuppDiffRatio = 1;
    gSuppDiffRatio = 0.1;

    StopWatch sw;
    TestRunAggregater tg;
    tg.Reset();
    sw.Restart();

    Supp S_new, S_old;
    vi seq_new(gK, 0), seq_old(gK, 0);
    int htp_flag = HTPOriginal(S_new, seq_new);
    //int dfs_flag = dfsHTPKai(S_new, seq_new);
    //int dfs_flag = dfsHTPKaiMaxDist(S_new, seq_new);
    int dfs_flag = dfsHTPKaiMaxDistLocal(S_new, seq_new);
    tg.test1 += sw.ElapsedUs();
    cout<<"result dfs_flag: "<< dfs_flag <<" HTP num: "<< gHTPCnt<<" time: "
        << tg.test1/1000 << endl; 
    return;
}

void benchmarkHTPKaiRand0315(){
    //* 400*800 problem k = 140 150 160 170 180 190 ... 230
    gM = 400, gN = 800; //gK = k;
    //gSuppDiffRatio = 1;
    gSuppDiffRatio = 0.1;
    int numTest = 100;
    
    for(gK = 140; gK <= 230; gK += 10){
        
        int cnt = 0;
        REP(test, numTest){
            clearGlobalRecord();
            genGaussMatAndShortSignal();
            Supp S_new, S_old;
            vi seq_new(gK, 0), seq_old(gK, 0);
            int htp_flag = HTPOriginal(S_new, seq_new);
            int dfs_flag = dfsHTPKaiMaxDist(S_new, seq_new);
            cnt += (dfs_flag == 0? 1: 0);
        }
        cout<<"K = "<<gK <<",  success = " << cnt <<endl;
    }
    return;
}

void testLsSequence(MatrixXd const &A, MatrixXd const &b){
    MatrixXd X = LsSequence(A, b);
    MatrixXd X_ref = LsSequenceRef(A, b);
    X -= X_ref;
    cout << "Test LsSequence: difference: " << X.norm()<<endl;
}

void testLsRecoverSmall(MatrixXd const &A, MatrixXd const &b, MatrixXd const &v, vi cols){
    //* A: original, b: original rhs, v: candidate for column replacement, cols: replaced columns
    HouseholderQR<MatrixXd> qr_base(A);
    MatrixXd A_test = A;
    REP(j, cols.size()){
        A_test.col(cols[j]) = v.col(j);
    }
    MatrixXd x_recover = LsRecover(A_test, b, cols, qr_base);
    MatrixXd x_ref = LsReference(A_test, b);
    x_recover -= x_ref;
    cout << "Test columns ";
    for(int j: cols)
        cout << j <<" ";
    cout << "replaced. Difference of x: " << x_recover.norm() <<endl;
}

void testQrRecoverOneColSmallInner(MatrixXd const &A, MatrixXd const &v, int j){
    //* QR factorization for base matrix A
    HouseholderQR<MatrixXd> qr(A);

    MatrixXd A_last(A); A_last.col(j) = v.col(0);
    MatrixXd v_last = v.col(0);
    MatrixXd R(10, 8);
    R = qr.matrixQR().triangularView<Upper>();
    v_last.applyOnTheLeft(qr.householderQ().transpose());
    R.col(j) = v_last;
    vi cols;
    GivensRotationSequence Gseq(100);
    QrRecoverOneCol(R, cols, j, Gseq);
    HouseholderQR<MatrixXd> qr_last(A_last); //* for reference
    MatrixXd R_ref = qr_last.matrixQR().triangularView<Upper>();
    cout << "=======R_ref======="<<endl; cout << R_ref << endl;
    cout << "=======R======="<<endl; cout << R << endl;
    
    R -= R_ref;
    cout << "Test column "<< j << " replaced. ";
    cout << "Difference of R: " << R.norm() <<endl;
    return;
}

void testQrRecoverSmallInner(MatrixXd const &A, MatrixXd const &v, vi cols){
    //* QR factorization for base matrix A
    HouseholderQR<MatrixXd> qr(A);

    MatrixXd A_last(A); 
    MatrixXd R(10, 8);
    MatrixXd v_last;
    R = qr.matrixQR().triangularView<Upper>();
    
    REP(j, cols.size()){
        A_last.col(cols[j]) = v.col(j);
        v_last = v.col(j);
        v_last.applyOnTheLeft(qr.householderQ().transpose());
        R.col(cols[j]) = v_last;
    }
    
    GivensRotationSequence Gseq(100);
    QrRecover(R, cols, Gseq);

    HouseholderQR<MatrixXd> qr_ref(A_last); //* for reference
    MatrixXd R_ref = qr_ref.matrixQR().triangularView<Upper>();
    cout << "=======R_ref======="<<endl; cout << R_ref << endl;
    cout << "=======R======="<<endl; cout << R << endl;
    
    R -= R_ref;
    cout << "Test columns ";
    for(int j: cols)
        cout << j <<" ";
    cout << "replaced. Difference of R: " << R.norm() <<endl;
    return;
}

void testQrRecoverSmall(){
    //* Base matrix
    MatrixXd A(10, 8);
    A << 
    2.1587,   -0.3880,   -0.0896,   -0.7525,    1.8769,    0.7739,   -0.0788,    1.4911,
   -1.6574,    1.1081,   -0.7964,    0.9815,    0.8387,    0.2732,    0.1348,   -0.2452,
   -0.3853,    0.2207,    0.9679,   -0.0960,    1.0489,    1.2282,   -2.1436,   -1.3513,
    1.2011,    0.4931,    1.3274,   -1.2675,   -0.8045,   -0.2840,    1.1084,   -0.3390,
   -1.2144,   -1.0815,    0.4037,   -1.0503,   -1.5747,    0.8066,   -0.0623,    0.0846,
    0.8664,    1.2181,    0.3562,   -0.4589,   -1.7962,   -0.3609,   -0.1248,    0.6676,
    2.4110,   -1.1709,   -0.4480,   -0.9662,    1.7454,    1.0210,    0.4721,    1.8396,
    0.7889,   -0.2369,    1.7156,    0.2490,   -0.9512,    0.9452,    1.5492,   -0.2247,
    0.1950,   -0.2788,   -0.0487,   -0.0264,   -1.1535,   -1.2915,   -0.6619,   -0.5003,
   -0.9998,    0.6505,    0.6901,   -1.4151,   -0.9421,   -1.1142,   -0.5234,    0.5920;
    //* vectors for exchange
    MatrixXd v(10, 3);
    v << 0.4308, -0.9412,  0.1989, -0.6054, -0.7239, -0.0560,  0.3643,  2.7442,  0.7178, -1.1607,
        -0.6066,  0.9648, -0.4463, -0.0177,  1.2225, -0.4122,  1.4584, -0.5695, -0.4413,  1.5557,
         0.5377,  1.8339, -2.2588,  0.8622,  0.3188, -1.3077, -0.4336,  0.3426,  3.5784,  2.7694;
    MatrixXd b(10, 1);
    b << 1, 2, 3, 4, 5, 6, 7, 8, 9, 0;
    
    //testQrRecoverOneColSmallInner(A, v, 7);
    //testQrRecoverOneColSmallInner(A, v, 0);
    testQrRecoverOneColSmallInner(A, v, 4);
    
    // vi cols{1, 5, 7};
    // testQrRecoverSmallInner(A, v, cols);
    // testLsRecoverSmall(A, b, v, cols);

    // testLsSequence(A, b);
    return;
}

void benchmarkLsRecover(int m, int n){
    //* random generator settings
    random_device rd;
    mt19937 gen(rd());  
    normal_distribution<double> dis(0, 1.0);
    //* declare random matrix
    MatrixXd A = Eigen::MatrixXd::Zero(m,n).unaryExpr([&](double dummy){return dis(gen);});
    MatrixXd b = Eigen::MatrixXd::Zero(m,1).unaryExpr([&](double dummy){return dis(gen);});
    MatrixXd V = Eigen::MatrixXd::Zero(m,n).unaryExpr([&](double dummy){return dis(gen);});
    MatrixXd A_test(A), x;
    int r = n/10;
    vi cols;
    StopWatch sw;
    TestRunAggregater tg;
    tg.Reset();
    HouseholderQR<MatrixXd> qr(A);
    int num_test = 50;
    //* 1. Reference 
    REP(_, num_test){
        sw.Restart();
        x = LsReference(A, b);
        tg.test1 += sw.ElapsedUs();
    }
    //* 2. Test all replaced columns are in the last 20% columns
    for(int j = n/10*8; j < n; j+=2){
        cols.emplace_back(j);
    }
    REP(j, cols.size()){
        A_test.col(cols[j]) = V.col(j);
    }
    REP(_, num_test){
        sw.Restart();
        x = LsRecover(A, b, cols, qr);
        tg.test2 += sw.ElapsedUs();
    }

    //* 3. Test all replaced columns are in the last 50% columns
    cols.clear();
    A_test = A;
    for(int j = n/10*5; j < n; j+=5){
        cols.emplace_back(j);
    }
    REP(j, cols.size()){
        A_test.col(cols[j]) = V.col(j);
    }
    REP(_, num_test){
        sw.Restart();
        x = LsRecover(A, b, cols, qr);
        tg.test3 += sw.ElapsedUs();
    }
    cout << "Benchmark LsRecover, m = " << m << ", n = " << n <<endl;
    cout << "Ref: " << (tg.test1/ num_test) << ", 10% cols changed in last 20%: " << (tg.test2 / num_test) << ", 10% cols changed in last 50% " << (tg.test3 / num_test) <<endl;

}

void benchmarkLsSequence(int m, int n){
    //* random generator settings
    random_device rd;
    mt19937 gen(rd());  
    normal_distribution<double> dis(0, 1.0);
    //* declare random matrix
    MatrixXd A = Eigen::MatrixXd::Zero(m,n).unaryExpr([&](double dummy){return dis(gen);});
    MatrixXd b = Eigen::MatrixXd::Zero(m,1).unaryExpr([&](double dummy){return dis(gen);});
    // cout<<"normal random matrix:\n"<<A<<endl;
    // JacobiSVD<MatrixXd> svd(A);
    // double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
    // cout << "conditional number: "<< cond<<endl;

    StopWatch sw;
    TestRunAggregater tg;
    tg.Reset();
    MatrixXd X(n, n);
    MatrixXd x;
    sw.Restart();
    // X = LsSequenceRef(A, b);
    tg.test1 += sw.ElapsedUs();

    int num_test = 50;
    REP(_, num_test){
        sw.Restart();
        X = LsSequence(A, b);
        tg.test2 += sw.ElapsedUs();

        sw.Restart();
        x = A.householderQr().solve(b);
        tg.test3 += sw.ElapsedUs();
    }
    cout << "Benchmark LsSequence, m = " << m << ", n = " << n <<endl;
    cout << "Ref: " << (tg.test1) << ", New: " << (tg.test2 / num_test) << ", One: " << (tg.test3 / num_test) <<endl;
}

int main()
{
    //testQrRecoverSmall();
    /*
    int m, n;
    cin >> m >> n;
    benchmarkLsSequence(m, n);
    benchmarkLsRecover(m, n);
    */
    testHTPKaiRand();
    //benchmarkHTPKaiRand0315();
    return 0;
}
