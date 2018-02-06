/*
 * @file   main.cpp
 * @author SenseTime Group Limited
 * @brief  Common header for SenseTime C API.
 * * Copyright (c) 2014-2015, SenseTime Group Limited. All Rights Reserved.
 * author Weicai Ye, SenseTime Group Limited, (yeweicai@sensetime.com)
 */
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/affine.hpp>

using namespace std;
using namespace cv;

typedef cv::Matx33f Mat3f;
typedef cv::Vec3f Vec3f;
typedef cv::Vec3i Vec3i;
typedef cv::Affine3f Affine3f;


/**
 *  \breif the camera intrinsics
 */
struct Intr {
    double fx, fy ,cx, cy;
    Intr();
    Intr(float fx, float fy, float cx, float cy);
    Intr operator()(int level_index) const;
};

Intr Intr::operator()(int level_index) const {
    int div = 1 << level_index;
    return (Intr(fx/div, fy/div, cx/div, cy/div));
}

struct Point {
    union {
        float data[4];
        struct {
            float x, y, z;
        };
    };
};

typedef Point Normal;

struct RGB {
    union {
        struct {
            unsigned char b, g, r;
        };
        int bgra;
    };
};
struct Frame {
    bool use_points;
    Mat depth_pyr;   //depth
    Mat points_pyr;  //cloud
    Mat normals_pyr; //normals
};

struct PixelRGB {
    unsigned char r, g, b;
};

struct KinFuParams {
    static KinFuParams default_params();
    static KinFuParams default_params_kinectfusion();


    int cols; //pixels
    int rows; //pixels

    Intr intr;

    int bilateral_kernel_size; //pixels
    float bilateral_sigma_spatial; //pixels
    float bilateral_sigma_depth; //meters
    float icp_truncate_depth_dist;
};

struct Projector {
    float2 f, c;
    Projector(){}
    Projector(float fx, float fy, float cx, float cy) {
    float2 operator()(const float3& p);
    }

};

struct Reprojector{
    Reprojector() {}
    Reprojector(float fx, float fy, float cx, float cy) {}
    float2 finv, c;
    float3 operator()(int x, int y, float z);
};

class TsdfVolume {
    TsdfVolume(const cv::Vec3i& dims);
    ~TsdfVolume();

    void create(const Vec3i& dims);

    Vec3i getDims() const;
    Vec3f getVoxelSize() const;

    Vec3f getSize() const;
    void setSize(const Vec3f& size);

    float getTruncDist() const;
    void setTruncDist(float distance);

    int getMaxWeight() const;
    void setMaxWeight(int weight);

    Affine3f getPose() const;
    void setPose(const Affine3f& pose);

    float getRaycastStepFactor() const;
    void setRaycastStepFactor(float factor);

    float getRaycastStepFactor() const;
    void setRaycastStepFactor(float factor);

    float getGradientDeltaFactor() const;
    float setGradientDeltaFactor(float factor) ;

    Vec3i getGridOrigin() const;
    void setGridOrigin(const Vec3i& origin);

    virtual void clear();

    virtual void applyAffine(const Affine3f& affine);
    virtual void integrate(const Dists& dists, const Affine3f& camera_pose, const Intr& intr);
    virtual void raycast(const Affine3f& camera_pose, const Intr& intr, Depth& depth, Normals& normals);
    virtual void raycast(const Affine3f& camera_pose, const Intr& intr, Cloud& points, Normals& normals);

    void swap();

private:
    float trunc_dist_;
    int max_weight_;
    Vec3i dims_;
    Vec3f size_;
    Affine3f pose_;

    float gradient_delta_factor_;
    float raycast_step_factor_;

};

struct TsdfIntegrator {
    Affine3f vol2cam;
    Projector proj;
    int2 dists_size;
    float tranc_dist_inv;

    void operator()(TsdfVolume& volume) {
        int x, y;
        if (x >= volume.dims_.x || y >= volume.dims_.y)
            return;

        float3 zstep = make_float3(vol2cam.R.data[0].z, vol2cam.R.data[1].z, vol2cam.R.data[2].z) * volume.voxel_size.z;
        float3 vx = make_float3(x * volume.voxel_size.x, y*volume.voxel_size.y, 0);
        float3 vc = voc2cam * vx;

        TsdfVolume::elem_type* vptr = volume.beg(x, y);

        for (int i = 0; i < volume.dims_.z; ++i, vc+=zstep, vptr = volume.zstep(vptr)) {
            float2 coo = proj(vc);

            if (coo.x < 0 || coo.y < 0 || coo.x >= dists_size.x || coo.y >= dists_size.y)
                continue;
            float Dp = tex2D(dists_size, coo.x, coo.y);

            if (Dp == 0 || vc.z <= 0)
                continue;

            if (sdf >= -volume.trunc_dist_) {
                float tsdf == fmin(1.f, sdf * tranc_dist_inv);
                int weight_prev;
                float tsdf_prev = unpack_tsdf(gmem:LdCs(vptr), weight_prev);
                float tsdf_new = __fdividef(__fmaf_rn(tsdf_prev, weight_prev, tsdf), weight_prev+1);
                int weight_new = min(weight_prev + 1, volume.max_weight_);

                gmem::StCs(pack_tsdf(tsdf_new, weight_new), vptr);
            }
        }
    }
    void integrate_kernel(const TsdfIntegrator integrator, TsdfVolume volume) {
        integrator(volume);
    }
};

struct TsfdfRaycaster {
    TsdfVolume volume;
    Affine3f aff;
    Mat3f Rinv;
    Vec3f volume_size;
    Reprojector reproj;

    float time_step;
    float3 gradient_delta;
    float3 voxel_size_inv;

    TsdfRaycaster(const TsdfVolume& volume, const Affine3f& aff, const Mat3f& Rinv, const Reprojector& _reproj) {
        int x = (p.x * voxel_size_inv.x);
        int y = (p.y * voxel_size_inv.y);
        int z = (p.z * voxel_size_inv.z);
        return unpack_tsdf(*volume(x, y, z));
    }

    void operator() (depth, normals) const {
        int x, y;
        if (x >= depth.cols || y >= depth.rows)
            return;
        const float qnan = numeric_limits<float>::quiet_NaN();

        depth(y, x) = 0;
        normals(y, x) = make_float4(qnan, qnan, qnan, qnan);

        float3 ray_org = aff.t;
        float3 ray_dir = normalized(aff.R * reproj(x, y, 1.f));

        float3 box_max = volume_size - volume.voxel_size;

        float tmin, tmax;
        intersect(ray_org, ray_dir, box_max, tmin, tmax);
        const float min_dist = 0.f;
        tmin = fmax(min_dist, tmin);
        if (tmin >= tmax)
            return;
        tmax -= time_step;
        float3 vstep = ray_dir * time_step;
        float3 next = ray_org + ray_dir * tmin;

        float tsdf_next = fetch_tsdf(next);
        for (float tcurr = tmin; tcurr < tmax; tcurr += time_step) {
            float tsdf_curr = tsdf_next;
            float3 curr = next;
            next += vstep;
            tsdf_next = fetch_tsdf(next);
            if (tsdf_curr < 0.f && tsdf_next > 0.f)
                break;
            if (tsdf_curr > 0.f && tsdf_next < 0.f) {
                float Ft   = interpolate(volume, curr * voxel_size_inv);
                float Ftdt = interpolate(volume, next * voxel_size_inv);

                float Ts = tcurr - __fdividef(time_step * Ft, Ftdt - Ft);

                float3 vertex = ray_org + ray_dir * Ts;
                float3 normal = compute_normal(vertex);

                if (!isnan(normal.x * normal.y * normal.z))
                {
                    normal = Rinv * normal;
                    vertex = Rinv * (vertex - aff.t);

                    normals(y, x) = make_float4(normal.x, normal.y, normal.z, 0);
                    depth(y, x) = static_cast<ushort>(vertex.z * 1000);
                }
                break;
            }

        }
    }


    void operator() (points, normals) {
        int x, y;
        if (x >= points.cols || y >= points.rows)
            return;

        const float qnan = numeric_limits<float>::quiet_NaN();

        points(y, x) = normals(y, x) = make_float4(qnan, qnan, qnan, qnan);

        float3 ray_org = aff.t;
        float3 ray_dir = normalized( aff.R * reproj(x, y, 1.f) );

        // We do subtract voxel size to minimize checks after
        // Note: origin of volume coordinate is placeed
        // in the center of voxel (0,0,0), not in the corener of the voxel!
        float3 box_max = volume_size - volume.voxel_size;

        float tmin, tmax;
        intersect(ray_org, ray_dir, box_max, tmin, tmax);

        const float min_dist = 0.f;
        tmin = fmax(min_dist, tmin);
        if (tmin >= tmax)
            return;

        tmax -= time_step;
        float3 vstep = ray_dir * time_step;
        float3 next = ray_org + ray_dir * tmin;

        float tsdf_next = fetch_tsdf(next);
        for (float tcurr = tmin; tcurr < tmax; tcurr += time_step) {
            float tsdf_curr = tsdf_next;
            float3 curr = next;
            next += vstep;

            tsdf_next = fetch_tsdf(next);
            if (tsdf_curr < 0.f && tsdf_next > 0.f)
                break;

            if (tsdf_curr > 0.f && tsdf_next < 0.f) {
                float Ft = interpolate(volume, curr * voxel_size_inv);
                float Ftdt = interpolate(volume, next * voxel_size_inv);

                float Ts = tcurr - __fdividef(time_step * Ft, Ftdt - Ft);

                float3 vertex = ray_org + ray_dir * Ts;
                float3 normal = compute_normal(vertex);

                if (!isnan(normal.x * normal.y * normal.z)) {
                    normal = Rinv * normal;
                    vertex = Rinv * (vertex - aff.t);

                    normals(y, x) = make_float4(normal.x, normal.y, normal.z, 0.f);
                    points(y, x) = make_float4(vertex.x, vertex.y, vertex.z, 0.f);
                }
                break;
            }
        }
    }

    float3 compute_normal(const float3& p) const
    {
        float3 n;

        float Fx1 = interpolate(volume, make_float3(p.x + gradient_delta.x, p.y, p.z) * voxel_size_inv);
        float Fx2 = interpolate(volume, make_float3(p.x - gradient_delta.x, p.y, p.z) * voxel_size_inv);
        n.x = __fdividef(Fx1 - Fx2, gradient_delta.x);

        float Fy1 = interpolate(volume, make_float3(p.x, p.y + gradient_delta.y, p.z) * voxel_size_inv);
        float Fy2 = interpolate(volume, make_float3(p.x, p.y - gradient_delta.y, p.z) * voxel_size_inv);
        n.y = __fdividef(Fy1 - Fy2, gradient_delta.y);

        float Fz1 = interpolate(volume, make_float3(p.x, p.y, p.z + gradient_delta.z) * voxel_size_inv);
        float Fz2 = interpolate(volume, make_float3(p.x, p.y, p.z - gradient_delta.z) * voxel_size_inv);
        n.z = __fdividef(Fz1 - Fz2, gradient_delta.z);

        return normalized (n);
    }
};


void TsdfVolume::integrate(const Dists &dists, const Affine3f &camera_pose, const Intr &intr) {
    Affine3f vol2cam = camera_pose.inv() * pose_;
    Projector proj(intr.fx, intr.fy, intr.cx, intr.cy);

    Vec3i dims = dims_;
    Vec3f vsz = getVoxelSize();
    Affine3f aff = vol2cam;

    TsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);

    TsdfIntegrator ti;
    ti.integrate_kernel(ti, volume);
}

texture<ushort, 2> dprev_tex;
texture<Normal, 2> nprev_tex;
texture<Point,  2> vprev_tex;

struct ComputeIcpHelper::Policy {
    enum {
        CAT_SIZE_X = 32,
        CTA_SIZE_Y = 8;
        CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y,
        B = 6, COLS = 6, ROWS = 6, DIAG = 6,
        UPPER_DIAG_MAT = (COLS * ROWS - DIAG) / 2 + DIAG,
        TOTAL = UPPER_DIAG_MAT + B,
        FINAL_REDUCE_CTA_SIZE = 256,
        FINAL_REDUCE_STRIDE = FINAL_REDUCE_CTA_SIZE
    };

    float2 ComputeIcpHelper::proj(const float3& p) const {
        float x = z * (u - c.x) * finv.x;
        float y = z * (v - c.y) * finv.y;
        return make_float3(x, y, z);
    }

//#if
    int ComputeICPHelper::find_coresp(int x, int y, float3& nd, float3& d, float3& s) const {
        int src_z = dcurr(y, x);
        if (src_z == 0)
            return 40;

        s = aff * reproj(x, y, src_z * 0.001f);

        float2 coo = proj(s);
        if (s.z <= 0 || coo.x < 0 || coo.y < 0 || coo.x >= cols || coo.y >= rows)
            return 80;

        int dst_z = tex2D(dprev_tex, coo.x, coo.y);
        if (dst_z == 0)
            return 120;

        d = reproj(coo.x, coo.y, dst_z * 0.001f);

        float dist2 = norm_sqr(s - d);
        if (dist2 > dist2_thres)
            return 160;

        float3 ns = aff.R * tr(ncurr(y, x));
        nd = tr(tex2D(nprev_tex, coo.x, coo.y));

        float cosine = fabs(dot(ns, nd));
        if (cosine < min_cosine)
            return 200;
        return 0;
    }

//#else
    int ComputeIcpHelper::find_coresp(int x, int y, float3& nd, float3& d, float3& s) const
    {
        s = tr(vcurr(y, x));
        if (isnan(s.x))
            return 40;

        s = aff * s;

        float2 coo = proj(s);
        if (s.z <= 0 || coo.x < 0 || coo.y < 0 || coo.x >= cols || coo.y >= rows)
            return 80;

        d = tr(tex2D(vprev_tex, coo.x, coo.y));
        if (isnan(d.x))
            return 120;

        float dist2 = norm_sqr(s - d);
        if (dist2 > dist2_thres)
            return 160;

        float3 ns = aff.R * tr(ncurr(y, x));
        nd = tr(tex2D(nprev_tex, coo.x, coo.y));

        float cosine = fabs(dot(ns, nd));
        if (cosine < min_cosine)
            return 200;
        return 0;
    }
};

class ProjectiveICP {
public:
    enum {
        MAX_PYRAMID_LEVELS = 4
    };

    typedef std::vector<Depth> Depthpyr;
    typedef std::vector<Cloud> PointsPyr;
    typedef std::vector<Normals> NormalsPyr;

    ProjectiveICP();
    virtual ~ProjectiveICP();

    float getDistThreshold() const;
    void setDistThreshold(float distance);

    float getAngleThreshold() const;
    void setAngleThreshold(float angle);

    void setIterationsNum(const std::vector<int>& iters);
    int getUsedLevelsNum() const;

    virtual bool estimateTransform(Affine3f& affine3f, const Intr& intr, const Frame& curr, const Frame& prev);
    virtual bool estimateTransform(Affine3f& affine3f, const Intr& intr, const DepthPyr& dcurr, const NormalsPyr ncurr, const DepthPyr& dprev, const NormalsPyr nprev);
    virtual bool estimateTransform(Affine3f& affine3f, const Intr& intr, const PointsPyr vcurr, const NormalsPyr ncurr, const PointsPyr vprev, const NormalesPyr nprev);

private:
    std::vector<int> iters_;
    float angle_thres_;
    float dist_thres_;
    float buffer_;
    struct StreamHelper;
    cv::Ptr<StreamHelper> shelp_;

};

bool ProjectiveICP::estimateTransform(Affine3f &affine3f, const Intr &intr, const Frame &curr, const Frame &prev) {
    CV_ASSERT("!Not implemented");
    return false;
}

bool ProjectiveICP::estimateTransform(Affine3f &affine3f, const Intr &intr, const DepthPyr &dcurr,
                                      const NormalsPyr ncurr, const DepthPyr &dprev, const NormalsPyr nprev) {

    const int LEVELS = getUsedLevelsNum();
    StreamHelper& sh = *shelp_;
    ComputeIcpHelper helper(dist_thres_, angle_thres_);
    affine = Affine3f::Identity();

    for (int level_index = LEVELS - 1; level_index >= 0; --level_index) {
        const Normals & n = (const device::Normals&) nprev[level_index];
        helper.rows = (float) n.rows();
        helper.cols = (float) n.cols();
        helper.setLevelIntr(level_index, intr.fx, intr.fy, intr.cx, intr.cy);
        helper.dcurr = dcurr[level_index];
        helper.ncurr = ncurr[level_index];

        for (int iter = 0; iter < iters_[level_index]; ++iter) {
            helper.aff = affine;
            helper(dprev[level_index], n, buffer_, sh, sh);
            StreamHelper::Vec6f b;
            StreamHelper::Mat6f A = sh.get(b);
            double det = cv::determinant(A);

            if (fabs(det) < 1e-15 || cv::viz::isNan(det)) {
                if (cv::viz::isNan(det))
                    cout << "qnan" << endl;
                return false;
            }
            StreamHelper::Vec6f r;
            cv::solve(A, b, r, cv::DECOMP_SVD);
            Affine3f Tinc(Vec3f(r.val), Vec3f(r.val + 3));
            affine = Tinc * affine;

        }

    }
    return true;
}

bool ProjectiveICP::estimateTransform(Affine3f &affine3f, const Intr &intr, const PointsPyr vcurr,
                                      const NormalsPyr ncurr, const PointsPyr vprev, const NormalesPyr nprev) {
    const int LEVELS = getUsedLevelsNum();
    StreamHelper& sh = *shelp_;

    device::ComputeIcpHelper helper(dist_thres_, angle_thres_);
    affine = Affine3f::Identity();

    for(int level_index = LEVELS - 1; level_index >= 0; --level_index)
    {
        const device::Normals& n = (const device::Normals& )nprev[level_index];
        const device::Points& v = (const device::Points& )vprev[level_index];

        helper.rows = (float)n.rows();
        helper.cols = (float)n.cols();
        helper.setLevelIntr(level_index, intr.fx, intr.fy, intr.cx, intr.cy);
        helper.vcurr = vcurr[level_index];
        helper.ncurr = ncurr[level_index];

        for(int iter = 0; iter < iters_[level_index]; ++iter)
        {
            helper.aff = device_cast<device::Aff3f>(affine);
            helper(v, n, buffer_, sh, sh);

            StreamHelper::Vec6f b;
            StreamHelper::Mat6f A = sh.get(b);

            //checking nullspace
            double det = cv::determinant(A);

            if (fabs (det) < 1e-15 || cv::viz::isNan (det))
            {
                if (cv::viz::isNan (det)) cout << "qnan" << endl;
                return false;
            }

            StreamHelper::Vec6f r;
            cv::solve(A, b, r, cv::DECOMP_SVD);

            Affine3f Tinc(Vec3f(r.val), Vec3f(r.val+3));
            affine = Tinc * affine;
        }
    }
    return true;
}

class KinFu {
public:
    typedef cv::Ptr<KinFu> Ptr;
    KinFu(const KinFuParams& params);

    const TsdfVolume& tsdf();

    bool operator() (Mat depth, Mat rgb);

private:
    int frame_counter_;
    KinFuParams params;
    Mat dists;  // dis
    Frame curr_, prev_;  //frame

    vector<Affine3f> poses;
//    Dists dists_;
    Cloud points;
    Normals normals;
    Depth depths_;
    TsdfVolume volume_;
    ProjectiveICP icp_;
};

void computeDists(Mat& depth, Mat& dists, Intr& intr) {
    int x, y;
//    float2 finv = (1.f/intr.fx, 1.f/intr.fy);
//    float2  c = (intr.cx, intr.cy);

    if (x < depth.cols || y < depth.rows) {
        float xl = (x - intr.cx) * (1.f / intr.fx);
        float yl = (y - intr.cy) * (1.f / intr.fy);
        float lambda = sqrtf(xl * xl + yl * yl + 1);
        dists[y][x] = depth[y][x] * lambda * 0.001f;
    }
}

void computePointNormals(const Intr& intr, const Depth& depth, Cloud& points, Normals& normals) {
    int x, y;
    Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);
    if (x >= depth.cols || y >= depth.rows) {
        return;
    }

    points(y, x) = normals(y, x) = make_float4(qnan, qnan, qnan, qnan);
    if (x >= depth.cols -1 || y >= depth.rows - 1) {
        return;
    }
    float z00 = depth(y, x) * 0.001f;
    float z01 = depth(y, x+1) * 0.001f;
    float z10 = depth(y+1, x) * 0.001f;

    if (z00 * z01 * z10 != 0) {
        float3 v00 = reproj(x, y, z00);
        float3 v01 = reproj(x+1, y, z01);
        float3 v10 = reproj(x, y+1, z10);

        float3 n = normalized(cross(v01 - v00, v10 - v00));
        normals(y, x) = make_float4(-n.x, -n.y, -n.z, 0.f);
        points(y, x) = make_float4(v00.x, v00.y, v00.z, 0.f);
    }

}

void computeNormalsAndMaskDepth(const Reprojector& reprojector, Depth& depth, Normals& normals) {
    //compute_normals
    int x, y;
    if (x >= depth.cols || y >= depth.rows) {
        return;
    }
    const float qnan = numeric_limits<float>::quiet_NaN();
    Normal n_out = make_float4(qnan, qnan, qnan, 0.f);
    if (x < depth.cols - 1 && y < depth.rows -1) {
        float z00 = depth(y, x) * 0.001f;
        float z01 = depth(y, x+1) * 0.001f;
        float z10 = depth(y+1, x) * 0.001f;
        if (z00 * z01 * z10 != 0) {
            float3 v00 = reproj(x, y, z00);
            float3 v01 = reproj(x + 1, y, z01);
            float3 v10 = reproj(x, y+1, z10);
            n_out = make_float4(-n.x, -n.y, -n.z, 0.f);
        }
    }
    normals(y, x) = n_out;

    //mask_depth
    if (x < depth.cols || y < depth.rows) {
        float4 n = normals(y, x);
        if (isnan(n.x)) {
            depth(y, x) = 0;
        }
    }
}

void depthTruncation(Depth& depth, float max_dist) {
    int x, y;
    if (x < depth.cols && y < depth.rows) {
        if (depth(y, x) > max_dist) {
            depth(y, x) = 0;
        }
    }
}

//depthBuildPyramid(D)
float ProjectiveICP::getDistThreshold() const {

}


bool KinFu::operator()(Mat depth, Mat rgb) {
    const KinFuParams& p = params;
    Intr intr = params.intr;

    const int LEVELS = icp_->getUsedLevelNum();

    computeDists(depth, dists, intr);
    bilateralFilter(depth, curr_.depth_pyr, p.bilateral_kernel_size, p.bilateral_sigma_spatial, p.bilateral_sigma_depth); //注意depth_pyr


    computePointNormals(intr, depth, point, normal);

    if (p.icp_truncate_depth_dist > 0) {
        depthTruncation(curr_.depth_pyr[0], p.icp_truncate_depth_dist);
    }

    for (int i = 1; i < LEVELS; i++) {
        depthBuildPyramid(curr_.depth_pyr[i - 1], curr_.depth_pyr[i], p.bilateral_sigma_depth);
    }

    for (int i = 0; i < LEVELS; i++) {
//#if
        computeNormalsAndMaskDepth(p.intr(i), curr_.depth_pyr[i], curr_.normals_pyr[i] );
//#else
        computePointNormals(p.intr(i), curr_.depth_pyr[i], curr_.points_pyr[i], curr_.normals_pyr[i]);
    }

    if (frame_counter_ == 0) {
        volume_->integrate(dists_, poses_.back(), p.intr);
        //
        curr_.depth_pyr.swap(prev_.depth_pyr);
        curr_.points_pyr.swap(prev_.points_pyr);
        curr_.normals_pyr.swap(prev_.normals_pyr);

        return ++frame_couter_, false;
    }

    //ICP
    Affine3f affine;
    bool ok = icp_.estimateTransform(affine, p.intr, curr_.depth_pyr, curr_.normals_pyr, prev_.depth_pyr, prev_.normals_pyr);
    bool ok = icp_.estimateTransform(affine, p.intr, curr_.points_pyr, curr_.normals_pyr, prev_.points_pyr, prev_.normals_pyr);

    if (!ok) {
        return false;
    }
    poses_.push_back(poses_.back() * affine);


    //volumn integration;
    //We do not integrate volume if camera does not move.
    float rnorm = (float)cv::norm(affine.rvec());
    float tnorm = (float)cv::norm(affine.translation());
    bool integrate = (rnorm + tnorm) / 2 >= p.tsdf_min_camera_movement;
    if (integrate) {
        volume_->integrate(dist_, poses_.back(), p.intr);
    }


    //Ray casting
    {
//#if
        volume_->raycast(poses_.back(), p.intr, prev_.depth_pyr[0], prev_.normals_pyr[0]);
        for (int i = 1; i < LEVELS; i++) {
            resizeDepthNormals(prev_.depth_pyr[i-1], prev_.normals_pyr[i-1], prev_.depth_pyr[i], prev_.normals_pyr[i]);

        }
//#else
        volume_->raycast(poses_.back(), p.intr, prev_.points_pyr[0], prev_.normals_pyr[0]);
        for (int i = 1; i < LEVELS; i++) {
            resizeDepthNormals(prev_.points_pyr[i-1], prev_.normals_pyr[i-1], prev_.points_pyr[i], prev_.normals_pyr[i]);
        }

    }

    return ++frame_counter_, true;
}

//设置相机内置参数、 双边过滤参数
//void setDefaultParams() {
//
//}


// 预处理，深度图像（双边滤波，达到去噪保边的作用）空间邻近度（高斯）和像素值相似度（距离）
// function bilateralFilter(Mat, Mat, int, float, float);
void recursiveBilateralFilter(const Mat &src, Mat& dst, int d, double sigmaColor, double sigmaSpace);


int main() {

    Mat image1 = imread("../data/meinv01.jpg");
    Mat image2 = imread("../data/meinv02.jpg");
    namedWindow("双边滤波[原图]");
    namedWindow("双边滤波[效果图]");

    imshow("双边滤波[原图]", image1);

    Mat out1;
    Mat out2;
    bilateralFilter(image1, out1, 25, 25*2, 25/2);
    bilateralFilter(image2, out2, 25, 25*2, 25/2);
    imshow("双边滤波[效果图]", out1);
    imwrite("../data/meinv01_bf.jpg", out1);
    imwrite("../data/meinv02_bf.jpg", out2);
    waitKey(0);
    std::cout << "Hello, World!" << std::endl;
    return 0;
}