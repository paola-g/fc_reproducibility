% load permutations
inDir = ['Z:\LabUsers\duboisjx\data\HCP\MRI\Results\FINAL\Finn\shen2013\',...
    'PMAT24_A_CR_Finn_shen2013_REST1_decon_Finn_all+MEG2_thr0.01'];


nPerm = 999;
rho     = nan(1,2);
rhoPerm = nan(nPerm,2);
for iPerm = 0:nPerm
    if mod(iPerm,50)==0
        fprintf('%d',iPerm);
    elseif mod(iPerm,10)==0
        fprintf('.');
    end
    try
        res = load(fullfile(inDir,sprintf('%04d',iPerm),'result.mat'),'rho_pos','rho_neg');
    catch
        continue
    end
    if iPerm==0
        rho(1) = res.rho_pos;
        rho(2) = res.rho_neg;
    else
        rhoPerm(iPerm,1) = res.rho_pos;
        rhoPerm(iPerm,2) = res.rho_neg;
    end
end
rhoPerm = rhoPerm(~isnan(sum(rhoPerm,2)),:);
size(rhoPerm,1)

% theoretical threshold
res    = load(fullfile(inDir,'0000','result.mat'),'pred_pos');
nSub   = length(res.pred_pos);
r      = linspace(-1,1,2000);
t      = r.*sqrt((nSub-2)./(1-r.^2));
p2     = 2*(1-tcdf(abs(t),nSub-2));
thThr  = [r([find(p2>=0.025,1,'first') find(p2>=0.025,1,'last')]);...
    r([find(p2>=0.005,1,'first') find(p2>=0.005,1,'last')])];

xs = linspace(-.25,.25,50);
figure;
i = 1;
% for i = 1:2
%     subplot(2,1,i);
% compute 95% & 99% interval from chance
empThr = [prctile(rhoPerm(:,i),[2.5 97.5]);...
    prctile(rhoPerm(:,i),[.5 99.5])];
Ns = hist(rhoPerm(:,i),xs);
bar(xs,Ns,'FaceColor','none','EdgeColor','k');
hold on;
v=axis;
% 99% empirical threshold
fill([empThr(2,:) empThr(2,end:-1:1)],[v(3) v(3) v(4) v(4)],'k','FaceColor','k','EdgeColor','none','FaceAlpha',.1);
% 95% empirical threshold
fill([empThr(1,:) empThr(1,end:-1:1)],[v(3) v(3) v(4) v(4)],'k','FaceColor','k','EdgeColor','none','FaceAlpha',.1);
% 95% theoretical threshold
line([repmat(thThr(1,1),2,1) repmat(thThr(1,end),2,1)],[v(3:4)' v(3:4)'],'Color','g','LineStyle','--','LineWidth',1);
line([repmat(thThr(2,1),2,1) repmat(thThr(2,end),2,1)],[v(3:4)' v(3:4)'],'Color','g','LineStyle','--','LineWidth',1);
line([0 0],v(3:4),'Color','k','LineStyle','-','LineWidth',1);
%     line(rho(i)*ones(1,2),v([3 4]),'Color','r')
% end
keyboard
