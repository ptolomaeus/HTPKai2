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

void testHTPSmall(){
    //* small input data
    /*
    mat A_in(8, 10);
    A_in <<  0.5377 ,  3.5784  , -0.1241 ,   0.4889 ,  -1.0689  , -0.1022 ,   1.0933  , -0.7697  ,  1.5442  , -0.1924,
            1.8339 ,  2.7694  ,  1.4897 ,   1.0347 ,  -0.8095  , -0.2414 ,   1.1093  ,  0.3714  ,  0.0859  ,  0.8886,
            -2.2588,  -1.3499 ,   1.4090,    0.7269,   -2.9443 ,   0.3192,   -0.8637 ,  -0.2256 ,  -1.4916 ,  -0.7648,
            0.8622 ,  3.0349  ,  1.4172 ,  -0.3034 ,   1.4384  ,  0.3129 ,   0.0774  ,  1.1174  , -0.7423  , -1.4023,
            0.3188 ,  0.7254  ,  0.6715 ,   0.2939 ,   0.3252  , -0.8649 ,  -1.2141  , -1.0891  , -1.0616  , -1.4224,
            -1.3077,  -0.0631 ,  -1.2075,   -0.7873,   -0.7549 ,  -0.0301,   -1.1135 ,   0.0326 ,   2.3505 ,   0.4882,
            -0.4336,   0.7147 ,   0.7172,    0.8884,    1.3703 ,  -0.1649,   -0.0068 ,   0.5525 ,  -0.6156 ,  -0.1774,
            0.3426 , -0.2050  ,  1.6302 ,  -1.1471 ,  -1.7115  ,  0.6277 ,   1.5326  ,  1.1006  ,  0.7481  , -0.1961;
    mat x_in(10, 1);
    x_in << 100, 20, 4, 0, 0, 0, 0, 0, 0, 0;
    mat b_in(8, 1);    
    */    
    mat A_in(16, 20), x_in(20, 1);
    A_in << 0.0808675427814830, -0.0309870393235676, -0.242319828952223, 0.303521068910549, 0.359338630078115, 0.344543983916911, -0.0192090413254405, 0.270557315522832, 0.182311413817556, -0.231371786783949, -0.0599203119414645, -0.0442629365059290, -0.113836803799456, -0.0436320812019505, -0.362257824899608, -0.373014185103337, 0.405993833606720, 0.323168343765350, 0.225910095133369, 0.248810231562230,
            0.275824509192221, 0.371835842838696, -0.183518576709739, 0.307965233465766, 0.0199962037374672, 0.0707834308202099, -0.450109409757722, -0.214144892424100, -0.533210868977102, -0.272516043531107, -0.172961302989392, 0.0624065735346029, 0.00865514438210809, -0.0392556302859855, -0.106940788052126, 0.326514852271069, -0.453072053737302, -0.209082159725410, -0.305451040256134, -0.200490774186164,
            -0.339740671753232, 0.351701932234830, -0.667488215311106, -0.239774132785065, -0.347093569491893, 0.0480195314551629, -0.102214409682663, 0.191934289080674, -0.0919559245186917, -0.321830039565087, -0.228485498534866, -0.161768329828868, 0.0192628946429087, -0.142990614515798, -0.0375065540958116, 0.122503545012509, 0.447272458762712, 0.0825706218333416, -0.00463567633704324, 0.130355028620558,
            0.129674723898473, 0.353738190058297, 0.326090092451980, 0.0214770432245295, -0.172733888452773, 0.385421163565457, -0.417895632054948, 0.0247768502983816, -0.214027584621359, -0.148490539390312, 0.515257313212897, 0.292863410523159, 0.310240789578248, 0.452105043547884, 0.0663992569505217, -0.0101430084589822, 0.0677221910745642, -0.303349522916734, -0.369317641400413, 0.0291916254486794,
            0.0479437179168276, 0.167608984087787, 0.0737227933429510, -0.337072784133456, -0.247030428193058, -0.195287763848316, 0.195683625982505, 0.286955860826712, -0.409833933763767, -0.557339515173582, 0.337690952379419, -0.462622376844440, 0.573479956455118, -0.235372931511458, -0.0628144372816874, 0.0638015911925836, 0.200632679957563, -0.424579987630106, 0.193390186885347, 0.239625590465037,
            -0.196682169075062, -0.301394669137412, -0.171147120564945, -0.309138889896437, 0.546952190557025, 0.169108740201677, -0.206780578094442, -0.391656622849160, 0.132008701285602, 0.268347932325619, 0.0627314961130170, 0.154591840857030, 0.175357012697786, -0.130036714143612, 0.106650747100072, -0.547283517500905, -0.333865472945937, 0.297069143094896, 0.163300147968828, 0.155527929954154,
            -0.0652141796188536, 0.179026291671239, 0.310655519863902, -0.00190156468470628, -0.143250765911004, 0.202721444066151, 0.0233068763071911, -0.0394868782464065, 0.0732799362369733, 0.144734281733218, -0.256428941876532, -0.288340710044895, -0.0787611213793404, -0.191368027906029, 0.0942574306858911, -0.0295625609971842, -0.118373016485860, 1.60123343566824e-05, 0.000220221439797368, 0.260458610828281,
            0.0515322523321539, 0.406914738671749, -0.388011814431073, 0.425500957899823, 0.174077720442443, -0.0591629569961672, -0.126794976412661, -0.241246715198960, 0.00870050457518445, -0.00557381101536101, -0.176539503740541, -0.113774561762476, 0.234800008326110, -0.315597278866791, -0.300810212381605, 0.560883056391453, -0.0557853655333818, -0.0176408492796977, -0.0134240554097403, -0.0382724098014520,
            0.538206905961731, 0.122030287349868, -0.0231790221703351, -0.213680743356531, -0.0447758524576661, 0.0523548927960600, 0.0706756057919367, 0.580825439799328, -0.346586375779200, -0.00967689724193817, -0.0360097009010465, 0.187802734645284, 0.0688138618751566, -0.0516689077441281, -0.228001227828782, 0.0343911754958062, 0.0848054241586254, 0.292668908013521, -0.471164965796361, -0.0427000567904597,
            0.416535717030994, 0.258264459452311, -0.0547376020884635, 0.103105125690718, 0.206779946374915, -0.283013908268273, -0.139787600058205, 0.164823522530219, 0.293004367654904, -0.222131312889027, 0.161434268903784, 0.353020885495636, -0.386745292953249, -0.0736628431399725, -0.178249013634688, 0.0144679145778731, -0.335076462863935, 0.190989961559836, 0.110135465726673, 0.292333939918615,
            -0.203029033865502, 0.181434100991990, 0.0723662272110636, -0.0626285273981696, -0.177980675893510, -0.278670749737927, 0.114089698305563, 0.275426338951162, 0.0910020394290496, 0.283503160900928, -0.271704316297457, -0.379706810276471, 0.356495091976684, 0.411243102362932, -0.122139027304701, -0.256731165405108, 0.0946195041495173, 0.112490317221887, -0.415478906280457, -0.616027933550723,
            0.456466064597713, -0.0757403458459540, 0.0709270623433175, 0.310209255820400, -0.326308463178103, 0.0254588135314321, 0.172162624710221, -0.211353616371658, -0.0777190715963642, -0.0370748230000740, -0.475249895947679, 0.428296465087511, 0.115321902129357, -0.0669312772506185, -0.0771040318169716, -0.0107752351327929, -0.243321985306541, 0.401601044688302, -0.439516829615607, -0.146369939092186,
            0.109104040176642, 0.0733517621851927, -0.196073857246908, -0.302354650207743, -0.330987358578577, 0.175330445956577, 0.398617518940705, -0.0935980396592374, 0.00594843031724175, -0.198855881739488, -0.295589100830817, 0.224277174874197, 0.0507670618566260, -0.286032473210970, 0.00299902313832172, 0.0812492902491760, 0.0132790791738114, 0.298663508136029, 0.0151478934345066, -0.368572815670517,
            -0.00948373497318466, -0.196509656294059, -0.00681282273115984, 0.00903886091388162, 0.113602887849986, 0.627639741853064, -0.0452021697690863, -0.0544211579841857, -0.0680854390095025, 0.376094701343815, 0.0680300541243429, -0.0230566579782798, 0.193508812955469, 0.430966995073150, -0.728570278459010, 0.149103214279532, 0.130875950406699, 0.0770158607587516, -0.179742549296658, -0.110979831517399,
            0.107500529791932, 0.221747915001278, -0.0373791374129863, 0.153396925193200, -0.0412752589038300, -0.161890738551016, -0.497921581421272, 0.219391746869266, -0.454832266375112, -0.0625544571889715, 0.0798289118424939, -0.0663244923984954, 0.0981752299421645, 0.331849162705400, -0.109920036575115, -0.130367269721169, 0.0656149270486681, -0.221755223226226, 0.0779798167675179, 0.188168256296364,
            -0.0308278119847564, -0.286314335064282, 0.142305291893579, 0.305560120696525, -0.0456217128852852, 0.0454754568551802, -0.195500421652031, -0.0555002207228396, -0.0742328651912429, -0.163928541033618, 0.0921342645277035, -0.0739296289095691, -0.353589693687912, -0.0617175285409473, 0.298830632674498, -0.0826856643837556, 0.217198451446293, -0.209289634973706, 0.128291150501342, 0.239526137090757;
    x_in << -1.01494364268014, -0.471069912683167, 0.137024874130050, -0.291863375753573, 0.301818555261006, 0.399930942955802, -0.929961558940129, 3, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    mat b_in = A_in*x_in;
    //* assign to global variables
    gA = A_in; gRhs = b_in;
    gM = A_in.rows(); gN = A_in.cols(); gK = 9;
    gRhsNorm = gRhs.norm();
    gResTol = gRhsNorm * gResTolRatio;
    //gSuppDiffRatio = 1;
    gSuppDiffRatio = 1;

    //* Original HTP start from empty support set
    Supp S_new, S_old;
    vi seq_new(gK, 0), seq_old(gK, 0);
    int htp_flag = HTPOriginal(S_new, seq_new);
    int dfs_flag = dfsHTPKai(S_new, seq_new);
    cout<<"result dfs_flag: "<< dfs_flag <<endl; 

    cout<<"htpflag: "<<htp_flag <<endl;
    printSupp(S_new);
    //* test makeSeed
    vector<Supp> arrS, candidS;
    vector<vi> arrSeq, candidSeq;
    int seed_flag = makeSeed(S_new, seq_new, arrS, arrSeq);
    

    while(!arrS.empty()){
        S_old = arrS.back();
        seq_old = arrSeq.back();
        arrS.pop_back(); arrSeq.pop_back();
        cout<<"Input seq: "; printVec(seq_old);
        htp_flag = HTPInner(S_old, seq_old, S_new, seq_new);
        if(htp_flag == 2){
            candidS.push_back(S_new);
            candidSeq.push_back(seq_new);
        }
        // cout<<"htp_flag: "<<htp_flag << " dist_fp: "<< calHamDistClosest(S_new) <<endl;
    }
    int farIdx = 0, maxDist = 0;
    REP(i, candidS.size()){
        int dist = calHamDistClosest(candidS[i]);
        if(dist >= maxDist){
            maxDist = dist;
            farIdx = i;
        }
    }
    REP(i, candidS.size()) gFixedPoint.insert(candidS[i]);

    S_new = candidS[farIdx]; seq_new = candidSeq[farIdx];
    // #################################################
    // SECOND ROUND
    // #################################################
    cout<<"S_new: "; printSupp(S_new);
    seed_flag = makeSeed(S_new, seq_new, arrS, arrSeq);
    while(!arrS.empty()){
        S_old = arrS.back();
        seq_old = arrSeq.back();
        arrS.pop_back(); arrSeq.pop_back();
        cout<<"Input seq: "; printVec(seq_old);
        htp_flag = HTPInner(S_old, seq_old, S_new, seq_new);
        if(htp_flag == 2){
            candidS.push_back(S_new);
            candidSeq.push_back(seq_new);
        }
        // cout<<"htp_flag: "<<htp_flag << " dist_fp: "<< calHamDistClosest(S_new) <<endl;
    }
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
    /* *
    int m, n;
    cin >> m >> n;
    benchmarkLsSequence(m, n);
    benchmarkLsRecover(m, n);
    */
    //testHTPSmall();
    testHTPKaiRand();
    //benchmarkHTPKaiRand0315();
    return 0;
}
