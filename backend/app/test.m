function toughness = test(size, individual)

format long;



timer=tic;

%size = int(size)
% materials
E = 1e3; poisson = 0.33; thickness = 1;
% matrix D
D=E/(1-poisson^2)*[1 poisson 0;poisson 1 0;0 0 (1-poisson)/2];
%model size
%Lx: Length in x direction
Lx=1.;
%Ly: Lenght in y direction
Ly=1.;
%critical shear strain of soft and stiff materials
criticalShearStrain=[1.0 0.2];
%Assign regular grid in x direction and y direction
%nodes_of_x_dir : grids of x direction
nodes_of_x_dir=double(size+1);
%nodes_of_y_dir : grids of y direction
nodes_of_y_dir=double(size+1);
%node_spacing: edge lenth
node_spacing_x=Lx/(nodes_of_x_dir-1);
node_spacing_y=Ly/(nodes_of_y_dir-1);
% numberElements: number of elements
numberElements=(nodes_of_x_dir-1)*(nodes_of_y_dir-1);
% numberNodes: number of nodes
numberNodes=nodes_of_y_dir*nodes_of_x_dir;
% nodesPerElement: nodes per elements
nodesPerElement=4;
%initialize connectivitive, second arg is the nodes per element
elementNodes=zeros(numberElements,nodesPerElement);
% coordinates and connectivities
arr_x=linspace(0,Lx,nodes_of_x_dir);
arr_y=linspace(0,Ly,nodes_of_y_dir);
%generate node coordinates
[x_coor,y_coor]=meshgrid(arr_x,arr_y);
nodeCoordinates=[x_coor(:) y_coor(:)];
%generate node connectitivity
for i=1:(nodes_of_y_dir-1)
    for j=1:(nodes_of_x_dir-1)
        elementNodes((i-1)*(nodes_of_y_dir-1)+j,:)=[(i-1)*nodes_of_y_dir+j i*nodes_of_y_dir+j ...
            i*nodes_of_y_dir+j+1 (i-1)*nodes_of_y_dir+j+1];
    end
end
%determine the crack with(flag, initial points and end points)
%flag: 1 for crack and 0 for w/o crack
%please note that in this verison, crack tip should be align with the
%meshes, i.e. the initial and end point of crack should be a node.
%Crack should be prependicular to the edge.
crack={1,[0.5 0],[0.5 0.25]};
%find all crack nodes
numberNodes=numberNodes+crack{3}(2)/node_spacing_y;
crackTipNode=find((nodeCoordinates(:,1)==crack{2}(1))&(nodeCoordinates(:,2)==crack{3}(2)));
crackTipElement=[find(elementNodes(:,1)==crackTipNode) find(elementNodes(:,2)==crackTipNode)];
crackNodes=find((nodeCoordinates(:,1)==crack{2}(1))&(nodeCoordinates(:,2)<crack{3}(2)));
outputNodes=find((nodeCoordinates(:,1)==crack{2}(1))&(nodeCoordinates(:,2)>=crack{3}(2)));
%duplicate nodes to the end
nodeCoordinates=[nodeCoordinates; nodeCoordinates(crackNodes,:)];
%insert crack nodes into connectivity
[is,pos]=ismember(elementNodes,crackNodes);
%find crackElements in the left edges
%4----3
%|    |
%1----2
%In here is edge 1-4
crackPos_1=find(is(:,1));
crackPos_2=find(is(:,4));
%Determine translation 
translation=nodes_of_x_dir*nodes_of_y_dir-min(crackNodes)+1;
%
elementNodes(crackPos_1,1)=elementNodes(crackPos_1,1)+translation;
elementNodes(crackPos_2,4)=elementNodes(crackPos_2,4)+translation;
% fid=fopen('mode.inp','w');
% fprintf(fid,'*Node \n');
% for i=1:length(nodeCoordinates)
%     fprintf(fid,'%d, %e, %e \n',i,nodeCoordinates(i,1),nodeCoordinates(i,2));
% end
% fprintf(fid,'*Element, type=CPS4\n');
% for i=1:length(elementNodes)
%     fprintf(fid,'%d, %d, %d, %d, %d \n',i,elementNodes(i,1),...
%         elementNodes(i,2),elementNodes(i,3),elementNodes(i,4));
% end
% fclose(fid);
% drawingMesh(nodeCoordinates,elementNodes,'Q4','k-o');
% GDof: global number of degrees of freedom
GDof=2*numberNodes;
% boundary conditions
% find boundary nodes
boundaryNodesLeftEdges=find((nodeCoordinates(:,1)==0));
boundaryNodesRightEdges=find((nodeCoordinates(:,1)==Lx));
%Assgine Dof and values, in this case, Mode II crack, we assign +dy in y direction
% on the right edge and -dy on the left edge.
dx=0.01*Lx;
dy=0.;
% prescribedDof=[2*boundaryNodesLeftEdges(:)-1 ;2*boundaryNodesLeftEdges(:) ;2*boundaryNodesRightEdges(:)-1;
%     2*boundaryNodesRightEdges(:)];
% 
% prescribedValue=[dx*ones(length(boundaryNodesLeftEdges),1); dy*ones(length(boundaryNodesLeftEdges),1) ;...
%     dx*ones(length(boundaryNodesLeftEdges),1);-dy*ones(length(boundaryNodesLeftEdges),1)];

prescribedDof=[2*boundaryNodesLeftEdges(:)-1; 2*boundaryNodesLeftEdges(1) ;2*boundaryNodesRightEdges(:)-1;
    2*boundaryNodesRightEdges(1)];

prescribedValue=[-dx*ones(length(boundaryNodesLeftEdges),1); dy ;...
    dx*ones(length(boundaryNodesLeftEdges),1);dy];


% GDof: global number of degrees of freedom
GDof=2*numberNodes;
% generate materail map according to mesh
materialRatio=0.125;
stiffnessReduction=0.1;
%geo=['all_geo_',num2str(materialRatio*100),'_percent.txt'];
%tough=['toughness',num2str(materialRatio*100),'.txt'];
%stren=['strength',num2str(materialRatio*100),'.txt'];
%geo_m=['soft_',num2str(materialRatio*100),'_percent.mat'];
%fid_1=fopen(geo,'w');
%fid_2=fopen(tough,'w');
%fid_3=fopen(stren,'w');
%load('materialMaps');
%materialMap=ones(numberElements,1);
% materialMap(28:29)=0.;
% materialMap(34:35)=0.;
geo_map = 1;
%load(geo_m,'geo_map');
%materialMap =[1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 1 1 1 1 0 0 1 1 1 1 0 1 1 1 1 0 1 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0]
%materialMap = [1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 1 1 1 1 0 0 1 1 1 1 0 1 1 1 1 0 1 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0];
materialMap = individual;
%materialMap1 = reshape(materialMap,[8,8]);
%materialMap1 = flipud(materialMap1.');
%materialMap = reshape(materialMap1,[64,1]);
for ii=1:1
    %temp=reshape(geo_map(ii,:),[4,8])';
    %symmetry
    %tmp=[temp temp(:,end:-1:1)];
    %skewsymmetry
    %tmp=[temp temp(end:-1:1,end:-1:1)];
    %materialMap=tmp(:);
    temp=[];
    tmp=[];

    % calculation of the system stiffness matrix
    stiffness=formStiffness2D(GDof,numberElements,...
    elementNodes,numberNodes,nodeCoordinates,D,thickness,materialMap,stiffnessReduction);
    force=zeros(GDof,1);
    % force(13)=1000;
    % force(15)=2000;
    % force(17)=1000;
    % solution
    displacements=solution(GDof,prescribedDof,prescribedValue,stiffness,force);
    % output displacements
    %outputDisplacements(displacements, numberNodes, GDof);

    % draw deformed shape
    % figure();
    % hold on;
    % % magnification factor (for 10% of model max length)
    % M = max(max(nodeCoordinates)-min(nodeCoordinates))/max(abs(displacements))*0.1;
    % % reform displacement vector to fit nodeCoordinates 
    % disp = vec2mat(displacements,2)*M;
    % drawingMesh(nodeCoordinates+disp,elementNodes,'Q4','r:');
    % print('deformation','-dpng','-r1000');
    % hold off;

    [ stress_gp_cell, stress_node_cell,strain_gp_cell,strain_node_cell]...
        = stressRecovery(displacements,numberElements, elementNodes,nodeCoordinates,D,thickness,materialMap,stiffnessReduction);
    % 
    % fprintf('Stresses on Gauss points\n')
    % fprintf('%7s%4s%12s%10s%10s\n','Element','gp','Sxx','Syy','Sxy')
    % for e=1:numberElements
    %     for q=1:4
    %         fprintf('%4i%7i%16.4e%12.4e%12.4e\n',e,q,stress_gp_cell{e}(q,:))
    %     end
    % end
    % 
    % fprintf('Stresses on Nodal points\n')
    % fprintf('%7s%6s%10s%10s%10s\n','Element','node','Sxx','Syy','Sxy')
    % for e=1:numberElements
    %     for q=1:4
    %         fprintf('%4i%7i%16.4e%12.4e%12.4e\n',e,q,stress_node_cell{e}(q,:))
    %     end
    % end

    strain_avg_node=zeros(numberNodes,3);
    cnt_node=zeros(numberNodes,1);
    for e=1:numberElements
        for q=1:4
            strain_avg_node(elementNodes(e,q),:)=...
                strain_avg_node(elementNodes(e,q),:)+strain_node_cell{e}(q,:);
            cnt_node(elementNodes(e,q))=cnt_node(elementNodes(e,q))+1;
        end
    end
    strain_avg_node=strain_avg_node./[cnt_node cnt_node cnt_node];

    stress_avg_node=zeros(numberNodes,3);
    cnt_node=zeros(numberNodes,1);
    for e=1:numberElements
        for q=1:4
            stress_avg_node(elementNodes(e,q),:)=...
                stress_avg_node(elementNodes(e,q),:)+stress_node_cell{e}(q,:);
            cnt_node(elementNodes(e,q))=cnt_node(elementNodes(e,q))+1;
        end
    end
    stress_avg_node=stress_avg_node./[cnt_node cnt_node cnt_node];

    %compute effective shear modulus
    shearModulusEffective=mean(stress_avg_node(:,1))/mean(strain_avg_node(:,1));
    %evaluate linear toughness
    stiffness=E*[stiffnessReduction 1];
    tipElementToughness=zeros(length(crackTipElement),2);
    %tipElementFactor=zeros(length(crackTipElement),1);
    for l=1:length(crackTipElement)
        strain_gp=strain_gp_cell{crackTipElement(l)}(l,1);
        tipElementToughness(l,1)=0.5*shearModulusEffective*(strain_gp\criticalShearStrain(...
            materialMap(crackTipElement(l))+1)*(2*dx/Lx))^2;
        tipElementToughness(l,2)=shearModulusEffective*abs(strain_gp\criticalShearStrain(...
            materialMap(crackTipElement(l))+1)*(2*dx/Lx));
    end
    %tipElementToughness
    [toughness,index]=min(tipElementToughness(:,1));
  
    strength=tipElementToughness(index,2);
    %toughness
    %strength
    
    %for k=1:length(materialMap)
       % fprintf(fid_1,'%d ',materialMap(k));
    %end
    %fprintf(fid_1,'\n');
    %fprintf(fid_2,'%12.8e\n',toughness);
    %fprintf(fid_3,'%12.8e\n',strength);
    fclose('all')
%         materialColor=zeros(length(materialMap),3);
%         for m=1:length(materialMap)
%             if materialMap(m)==0
%                 materialColor(m,:)=[0 0 0];
%             else
%                 materialColor(m,:)=[1,0,0.4980];
%             end
%         end

%         M = 1;
%         node_displacement=[displacements(1:2:end) displacements(2:2:end) ];
%         figure();
%         hold on;
%         for i=1:length(materialColor)
%             fill(nodeCoordinates(elementNodes(i,:),1),nodeCoordinates(elementNodes(i,:),2),materialColor(i,:))
%         end
%         hold off;

    draw=0;
    if(draw)
        materialColor=zeros(length(materialMap),3);
        for i=1:length(materialMap)
            if materialMap(i)==0
                materialColor(i,:)=[0.0588    0.9490    0.9490];
            else
                materialColor(i,:)=[0 0 0];
            end
        end

        M = 1;
        node_displacement=[displacements(1:2:end) displacements(2:2:end) ];
        figure();
        hold on;
        for i=1:length(materialColor)
            fill(nodeCoordinates(elementNodes(i,:),1)+node_displacement(elementNodes(i,:),1)...
            ,nodeCoordinates(elementNodes(i,:),2)+node_displacement(elementNodes(i,:),2)...
            ,materialColor(i,:))
        end
        axis equal;
        hold off;

        figure();
        patch('Faces',elementNodes,'Vertices', (nodeCoordinates+node_displacement*M),...
            'FaceColor','interp','FaceVertexCData',node_displacement(:,1),...
            'CDataMapping','scaled');
        colormap(jet);
        title('X displacement');
        xlabel('x (mm)');
        ylabel('y (mm)');
        axis equal;
        colorbar;
        box on;
        print('displacement_x','-dpng','-r1000');

        figure();
        patch('Faces',elementNodes,'Vertices', (nodeCoordinates+node_displacement*M),...
            'FaceColor','interp','FaceVertexCData',node_displacement(:,2),...
            'CDataMapping','scaled');
        colormap(jet);
        title('Y displacement');
        xlabel('x (mm)');
        ylabel('y (mm)');
        axis equal;
        colorbar;
        box on;
        print('displacement_y','-dpng','-r1000');

        figure;
        patch('Faces',elementNodes,'Vertices', (nodeCoordinates+node_displacement*M),...
            'FaceColor','interp','FaceVertexCData',stress_avg_node(:,1),...
            'CDataMapping','scaled');
        colormap(jet);
        title('\sigma_x_x');
        xlabel('x (mm)');
        ylabel('y (mm)');
        axis equal;
        colorbar;
        box on;
        print('stress11','-dpng','-r1000');
        hold off;

        figure;
        patch('Faces',elementNodes,'Vertices', (nodeCoordinates+node_displacement*M),...
            'FaceColor','interp','FaceVertexCData',stress_avg_node(:,2),...
            'CDataMapping','scaled');
        colormap(jet);
        title('\sigma_y_y');
        xlabel('x (mm)');
        ylabel('y (mm)');
        axis equal;
        colorbar;
        box on;
        print('stress22','-dpng','-r1000');
        hold off;

        figure;
        patch('Faces',elementNodes,'Vertices', (nodeCoordinates+node_displacement*M),...
            'FaceColor','interp','FaceVertexCData',stress_avg_node(:,3),...
            'CDataMapping','scaled');
        colormap(jet);
        title('\sigma_x_y');
        xlabel('x (mm)');
        ylabel('y (mm)');
        axis equal;
        colorbar;
        box on;
        print('shear_stress','-dpng','-r1000');
        hold off;
    end
end


