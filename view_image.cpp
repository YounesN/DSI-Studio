#include "view_image.h"
#include "ui_view_image.h"
#include "libs/gzip_interface.hpp"
#include <QPlainTextEdit>

view_image::view_image(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::view_image)
{
    ui->setupUi(this);
    ui->info->setColumnWidth(0,120);
    ui->info->setColumnWidth(1,200);
    ui->info->setHorizontalHeaderLabels(QStringList() << "Header" << "Value");
    ui->view->setScene(&source);
    connect(ui->slice_pos,SIGNAL(valueChanged(int)),this,SLOT(update_image()));
    connect(ui->contrast,SIGNAL(valueChanged(int)),this,SLOT(update_image()));
    connect(ui->brightness,SIGNAL(valueChanged(int)),this,SLOT(update_image()));
    source_ratio = 2.0;
    ui->tabWidget->setCurrentIndex(0);


    qApp->installEventFilter(this);
}

view_image::~view_image()
{
    qApp->removeEventFilter(this);
    delete ui;
}
bool view_image::eventFilter(QObject *obj, QEvent *event)
{
    if (event->type() != QEvent::MouseMove || obj->parent() != ui->view)
        return false;
    QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
    QPointF point = ui->view->mapToScene(mouseEvent->pos().x(),mouseEvent->pos().y());
    image::vector<3,float> pos;
    pos[0] =  ((float)point.x()) / source_ratio - 0.5;
    pos[1] =  ((float)point.y()) / source_ratio - 0.5;
    pos[2] = ui->slice_pos->value();
    if(!data.geometry().is_valid(pos))
        return true;
    ui->info_label->setText(QString("(%1,%2,%3) = %4").arg(pos[0]).arg(pos[1]).arg(pos[2]).arg(data.at(pos[0],pos[1],pos[2])));
    return true;
}

bool view_image::open(QString file_name)
{
    gz_nifti nifti;
    image::io::dicom dicom;
    image::io::bruker_2dseq seq;
    gz_mat_read mat;
    data.clear();
    float vs[3];
    QString info;
    begin_prog("loading...");
    check_prog(0,1);
    if(nifti.load_from_file(file_name.toLocal8Bit().begin()))
    {
        nifti >> data;
        image::flip_xy(data);
        nifti.get_voxel_size(vs);
        info = QString("sizeof_hdr=%1\ndim_info=%2\n").
                arg(nifti.nif_header.sizeof_hdr).
                arg((int)nifti.nif_header.dim_info);
        for(unsigned int i = 0;i < 8;++i)
            info +=  QString("dim[%1]=%2\n").
                    arg(i).arg(nifti.nif_header.dim[i]);
        info += QString("intent_p1=%1\n").arg(nifti.nif_header.intent_p1);
        info += QString("intent_p2=%1\n").arg(nifti.nif_header.intent_p2);
        info += QString("intent_p3=%1\n").arg(nifti.nif_header.intent_p3);
        info += QString("intent_code=%1\n").arg(nifti.nif_header.intent_code);
        info += QString("datatype=%1\n").arg(nifti.nif_header.datatype);
        info += QString("bitpix=%1\n").arg(nifti.nif_header.bitpix);
        info += QString("slice_start=%1\n").arg(nifti.nif_header.slice_start);

        for(unsigned int i = 0;i < 8;++i)
            info +=  QString("pixdim[%1]=%2\n").
                    arg(i).arg(nifti.nif_header.pixdim[i]);

        info += QString("vox_offset=%1\n").arg(nifti.nif_header.vox_offset);
        info += QString("scl_slope=%1\n").arg(nifti.nif_header.scl_slope);
        info += QString("scl_inter=%1\n").arg(nifti.nif_header.scl_inter);
        info += QString("slice_end=%1\n").arg(nifti.nif_header.slice_end);
        info += QString("slice_code=%1\n").arg((int)nifti.nif_header.slice_code);
        info += QString("xyzt_units=%1\n").arg((int)nifti.nif_header.xyzt_units);
        info += QString("scl_inter=%1\n").arg(nifti.nif_header.scl_inter);
        info += QString("cal_max=%1\n").arg(nifti.nif_header.cal_max);
        info += QString("cal_min=%1\n").arg(nifti.nif_header.cal_min);
        info += QString("slice_duration=%1\n").arg(nifti.nif_header.slice_duration);
        info += QString("toffset=%1\n").arg(nifti.nif_header.toffset);
        info += QString("descrip=%1\n").arg(nifti.nif_header.descrip);
        info += QString("aux_file=%1\n").arg(nifti.nif_header.aux_file);
        info += QString("qform_code=%1\n").arg(nifti.nif_header.qform_code);
        info += QString("sform_code=%1\n").arg(nifti.nif_header.sform_code);
        info += QString("quatern_b=%1\n").arg(nifti.nif_header.quatern_b);
        info += QString("quatern_c=%1\n").arg(nifti.nif_header.quatern_c);
        info += QString("quatern_d=%1\n").arg(nifti.nif_header.quatern_d);
        info += QString("qoffset_x=%1\n").arg(nifti.nif_header.qoffset_x);
        info += QString("qoffset_y=%1\n").arg(nifti.nif_header.qoffset_y);
        info += QString("qoffset_z=%1\n").arg(nifti.nif_header.qoffset_z);

        for(unsigned int i = 0;i < 4;++i)
            info +=  QString("srow_x[%1]=%2\n").
                    arg(i).arg(nifti.nif_header.srow_x[i]);
        for(unsigned int i = 0;i < 4;++i)
            info +=  QString("srow_y[%1]=%2\n").
                    arg(i).arg(nifti.nif_header.srow_y[i]);
        for(unsigned int i = 0;i < 4;++i)
            info +=  QString("srow_z[%1]=%2\n").
                    arg(i).arg(nifti.nif_header.srow_z[i]);

        info += QString("intent_name=%1\n").arg(nifti.nif_header.intent_name);


    }
    else
        if(dicom.load_from_file(file_name.toLocal8Bit().begin()))
        {
            dicom >> data;
            dicom.get_voxel_size(vs);
            std::string info_;
            dicom >> info_;
            info = info_.c_str();

        }
        else
            if(mat.load_from_file(file_name.toLocal8Bit().begin()))
            {
                mat >> data;
                mat.get_voxel_size(vs);
                for(unsigned int index = 0;index < mat.size();++index)
                {
                    std::string data;
                    mat[index].get_info(data);
                    info += QString("%1 [%2x%3]=%4\n").arg(mat[index].get_name().c_str()).
                            arg(mat[index].get_rows()).
                            arg(mat[index].get_cols()).
                            arg(data.c_str());
                }
            }
            else
            if(seq.load_from_file(file_name.toLocal8Bit().begin()))
            {
                seq >> data;
                seq.get_voxel_size(vs);
            }
    check_prog(0,0);
    QStringList list = info.split("\n");
    ui->info->clear();
    ui->info->setRowCount(list.size());
    for(unsigned int row = 0;row < list.size();++row)
    {
        QString line = list[row];
        QStringList value_list = line.split("=");
        ui->info->setItem(row,0, new QTableWidgetItem(value_list[0]));
        if(value_list.size() > 1)
            ui->info->setItem(row,1, new QTableWidgetItem(value_list[1]));
    }
    ui->info->selectRow(0);

    if(!data.empty())
    {
        ui->slice_pos->setRange(0,data.depth()-1);
        ui->slice_pos->setValue(data.depth() >> 1);
        ui->image_info->setText(QString("width:%1 height:%2 depth:%3 resolution:%4 x %5 x %6").
                                arg(data.width()).
                                arg(data.height()).
                                arg(data.depth()).
                                arg(vs[0]).
                                arg(vs[1]).
                                arg(vs[2]));
        update_image();
    }
    return !data.empty();
}
void view_image::update_image(void)
{
    if(data.empty())
        return;
    image::basic_image<float,2> tmp(image::geometry<2>(data.width(),data.height()));
    unsigned int offset = ui->slice_pos->value()*tmp.size();

    std::copy(data.begin() + offset,
              data.begin() + offset + tmp.size(),tmp.begin());
    max_source_value = std::max<float>(max_source_value,*std::max_element(tmp.begin(),tmp.end()));
    if(max_source_value + 1.0 != 1.0)
        image::divide_constant(tmp.begin(),tmp.end(),max_source_value/255.0);

    float mean_value = image::mean(tmp.begin(),tmp.end());
    image::minus_constant(tmp.begin(),tmp.end(),mean_value);
    image::multiply_constant(tmp.begin(),tmp.end(),ui->contrast->value());
    image::add_constant(tmp.begin(),tmp.end(),mean_value+ui->brightness->value()*25.5);

    image::upper_lower_threshold(tmp.begin(),tmp.end(),tmp.begin(),0.0f,255.0f);


    buffer.resize(image::geometry<2>(data.width(),data.height()));
    std::copy(tmp.begin(),tmp.end(),buffer.begin());

    source.setSceneRect(0, 0, data.width()*source_ratio,data.height()*source_ratio);
    source_image = QImage((unsigned char*)&*buffer.begin(),data.width(),data.height(),QImage::Format_RGB32).
                    scaled(data.width()*source_ratio,data.height()*source_ratio);
    source.clear();
    source.addRect(0, 0, data.width()*source_ratio,data.height()*source_ratio,QPen(),source_image);
}

void view_image::on_zoom_in_clicked()
{
     source_ratio *= 1.1;
     update_image();
}

void view_image::on_zoom_out_clicked()
{
    source_ratio *= 0.9;
    update_image();
}
